"""Specialized natural-language intent classifier for music listening requests.

Trains a scikit-learn TF-IDF + LogisticRegression pipeline on a committed CSV
dataset and turns a free-text request into a guarded intent prediction. This
module never selects songs; it only decides which listening intent (if any)
a request expresses, so the deterministic recommender can be driven by a
structured profile instead of raw text.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("text", "intent")
VALID_INTENTS = frozenset(
    {"workout", "study", "relax", "party", "sleep", "mood_boost", "out_of_scope"}
)
OUT_OF_SCOPE_INTENT = "out_of_scope"

MIN_CONFIDENCE = 0.35
MIN_MARGIN = 0.05
MIN_INPUT_LENGTH = 3
MAX_INPUT_LENGTH = 500

GUARDRAIL_MESSAGE = (
    "I could not confidently identify a music-listening intent. Please "
    "mention an activity, mood, or listening goal such as workout, "
    "studying, relaxing, partying, sleeping, or improving your mood."
)


@dataclass
class IntentPrediction:
    """Typed result of running the intent classifier on one request."""

    intent: Optional[str]
    confidence: float
    second_confidence: float
    margin: float
    accepted: bool
    reason: str
    normalized_text: str


def _default_training_data_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "intent_training_data.csv"


def load_training_data(csv_path: Optional[Path] = None) -> tuple[list[str], list[str]]:
    """Load and validate the intent training dataset.

    Raises FileNotFoundError if the CSV is missing, ValueError if the header
    does not match the required columns exactly, or if any row has blank
    text or an unrecognized intent label.
    """
    path = csv_path if csv_path is not None else _default_training_data_path()
    if not path.is_file():
        raise FileNotFoundError(f"Training data file not found: {path}")

    texts: list[str] = []
    intents: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(
                f"Training data must have columns {REQUIRED_COLUMNS}, "
                f"found {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            text = (row["text"] or "").strip()
            intent = (row["intent"] or "").strip()

            if not text:
                raise ValueError(f"Blank training text at row {row_number}")
            if intent not in VALID_INTENTS:
                raise ValueError(
                    f"Unknown intent label '{intent}' at row {row_number}; "
                    f"expected one of {sorted(VALID_INTENTS)}"
                )

            texts.append(text)
            intents.append(intent)

    if not texts:
        raise ValueError(f"Training data file is empty: {path}")

    return texts, intents


def build_pipeline() -> Pipeline:
    """Construct the untrained TF-IDF + LogisticRegression pipeline.

    C=10.0 is set because the default C=1.0 under this dataset's size and
    7-way class_weight="balanced" split keeps predict_proba nearly uniform
    even for unambiguous, in-vocabulary requests, which then fails the
    MIN_CONFIDENCE guardrail for cases that should clearly be accepted.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                    C=10.0,
                ),
            ),
        ]
    )


class IntentClassifier:
    """Guardrailed wrapper around a trained TF-IDF + LogisticRegression pipeline."""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self._csv_path = csv_path
        self._pipeline: Optional[Pipeline] = None
        self._classes: list[str] = []

    def train(self) -> "IntentClassifier":
        """Load the training CSV and fit the classification pipeline deterministically."""
        logger.info("classifier training started")
        texts, intents = load_training_data(self._csv_path)
        pipeline = build_pipeline()
        pipeline.fit(texts, intents)

        self._pipeline = pipeline
        self._classes = list(pipeline.named_steps["classifier"].classes_)
        logger.info("classifier training completed")
        logger.info("number of training examples: %d", len(texts))
        return self

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def predict(self, text: object) -> IntentPrediction:
        """Classify one natural-language request and apply guardrail rules.

        Raises TypeError if text is not a string. RuntimeError if called
        before train(). Never raises for empty, too-short, too-long, or
        low-confidence input -- those are rejected safely instead.
        """
        if self._pipeline is None:
            raise RuntimeError("IntentClassifier must be trained before predict() is called")
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        normalized_text = text.strip()

        if not normalized_text:
            return self._reject(
                normalized_text,
                "Request was empty or contained only whitespace.",
            )
        if len(normalized_text) < MIN_INPUT_LENGTH:
            return self._reject(
                normalized_text,
                "Request was too short or ambiguous to interpret.",
            )
        if len(normalized_text) > MAX_INPUT_LENGTH:
            return self._reject(
                normalized_text,
                f"Request was too long (max {MAX_INPUT_LENGTH} characters).",
            )

        probabilities = self._pipeline.predict_proba([normalized_text])[0]
        ranked = sorted(zip(self._classes, probabilities), key=lambda pair: pair[1], reverse=True)
        top_intent, top_confidence = ranked[0]
        second_confidence = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_confidence - second_confidence

        if top_intent == OUT_OF_SCOPE_INTENT:
            logger.info(
                "request rejected: predicted intent=%s confidence=%.3f",
                top_intent,
                top_confidence,
            )
            return IntentPrediction(
                intent=top_intent,
                confidence=float(top_confidence),
                second_confidence=float(second_confidence),
                margin=float(margin),
                accepted=False,
                reason=GUARDRAIL_MESSAGE,
                normalized_text=normalized_text,
            )

        if top_confidence < MIN_CONFIDENCE:
            logger.info(
                "request rejected: low confidence intent=%s confidence=%.3f",
                top_intent,
                top_confidence,
            )
            return IntentPrediction(
                intent=top_intent,
                confidence=float(top_confidence),
                second_confidence=float(second_confidence),
                margin=float(margin),
                accepted=False,
                reason=GUARDRAIL_MESSAGE,
                normalized_text=normalized_text,
            )

        if margin < MIN_MARGIN:
            logger.info(
                "request rejected: ambiguous margin intent=%s margin=%.3f",
                top_intent,
                margin,
            )
            return IntentPrediction(
                intent=top_intent,
                confidence=float(top_confidence),
                second_confidence=float(second_confidence),
                margin=float(margin),
                accepted=False,
                reason=GUARDRAIL_MESSAGE,
                normalized_text=normalized_text,
            )

        logger.info(
            "request accepted: predicted intent=%s confidence=%.3f",
            top_intent,
            top_confidence,
        )
        return IntentPrediction(
            intent=top_intent,
            confidence=float(top_confidence),
            second_confidence=float(second_confidence),
            margin=float(margin),
            accepted=True,
            reason="",
            normalized_text=normalized_text,
        )

    def _reject(self, normalized_text: str, reason: str) -> IntentPrediction:
        logger.info("request rejected before inference: %s", reason)
        return IntentPrediction(
            intent=None,
            confidence=0.0,
            second_confidence=0.0,
            margin=0.0,
            accepted=False,
            reason=GUARDRAIL_MESSAGE,
            normalized_text=normalized_text,
        )
