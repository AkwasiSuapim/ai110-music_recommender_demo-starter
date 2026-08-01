"""Deterministic keyword baseline for intent classification.

This baseline exists only to give the automated evaluation harness a simple,
transparent point of comparison against the trained TF-IDF + LogisticRegression
classifier in intent_classifier.py. It is never used by the main application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

KEYWORDS: dict[str, tuple[str, ...]] = {
    "workout": (
        "workout", "gym", "exercise", "cardio", "run", "running", "sprint",
        "lift", "lifting", "weights", "training", "train", "hiit", "jog",
        "jogging", "pump", "pumped", "boxing", "spin class", "fitness",
        "leg day",
    ),
    "study": (
        "study", "studying", "homework", "focus", "focused", "concentrate",
        "concentration", "reading", "read", "coding", "code", "exam",
        "thesis", "research", "assignment", "essay", "textbook", "lab report",
    ),
    "relax": (
        "relax", "relaxing", "calm", "chill", "unwind", "destress",
        "de-stress", "soothing", "mellow", "peaceful", "decompress",
        "tranquil", "laid back", "easygoing", "easy going", "couch", "sofa",
    ),
    "party": (
        "party", "dance", "dancing", "celebration", "celebrate", "celebrating",
        "festive", "birthday", "gathering", "hype", "crowd", "dancefloor",
        "dance floor", "hosting", "social", "cookout", "new year",
    ),
    "sleep": (
        "sleep", "asleep", "bedtime", "bed", "dream", "dreamy", "doze",
        "drift off", "nap", "night", "restful", "tuck", "midnight",
    ),
    "mood_boost": (
        "uplifting", "uplift", "cheer", "cheerful", "happy", "mood",
        "encouragement", "encouraging", "positive", "hopeful", "hope",
        "joyful", "joy", "motivate", "motivated", "inspiring", "smile",
        "blues", "down", "sad", "funk", "gray",
    ),
    "out_of_scope": (
        "weather", "recipe", "python", "javascript", "password", "capital",
        "email", "traffic", "airport", "flight", "translate", "translation",
        "tire", "engine", "timer", "lights", "population", "stock market",
        "gift", "movie", "movies", "joke", "boil", "egg", "mountain", "moon",
        "exchange rate", "bookshelf", "grammar", "declutter", "paint",
        "compound interest", "basil",
    ),
}

OUT_OF_SCOPE_INTENT = "out_of_scope"

REJECT_NO_MATCH = "No keyword matched any known intent."
REJECT_TIE = "Multiple intents tied for the top keyword score."
REJECT_OUT_OF_SCOPE = "Keywords matched only the out-of-scope vocabulary."


@dataclass
class KeywordPrediction:
    """Prediction produced by the deterministic keyword baseline."""

    intent: Optional[str]
    accepted: bool
    reason: str
    normalized_text: str
    scores: dict[str, int]


def _count_matches(normalized_text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in normalized_text)


def predict_intent(text: object) -> KeywordPrediction:
    """Classify one request using plain keyword counting, for comparison only."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    normalized_text = text.strip().lower()

    scores = {
        intent: _count_matches(normalized_text, keywords)
        for intent, keywords in KEYWORDS.items()
    }

    top_score = max(scores.values())
    if top_score == 0:
        return KeywordPrediction(
            intent=None,
            accepted=False,
            reason=REJECT_NO_MATCH,
            normalized_text=normalized_text,
            scores=scores,
        )

    top_intents = [intent for intent, score in scores.items() if score == top_score]
    if len(top_intents) > 1:
        return KeywordPrediction(
            intent=None,
            accepted=False,
            reason=REJECT_TIE,
            normalized_text=normalized_text,
            scores=scores,
        )

    top_intent = top_intents[0]
    if top_intent == OUT_OF_SCOPE_INTENT:
        return KeywordPrediction(
            intent=top_intent,
            accepted=False,
            reason=REJECT_OUT_OF_SCOPE,
            normalized_text=normalized_text,
            scores=scores,
        )

    return KeywordPrediction(
        intent=top_intent,
        accepted=True,
        reason="",
        normalized_text=normalized_text,
        scores=scores,
    )
