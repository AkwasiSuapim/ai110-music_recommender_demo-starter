import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from src.ai.intent_classifier import (
    IntentClassifier,
    MAX_INPUT_LENGTH,
    MIN_INPUT_LENGTH,
    OUT_OF_SCOPE_INTENT,
    VALID_INTENTS,
    load_training_data,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = REPO_ROOT / "data" / "intent_training_data.csv"
EVALUATION_JSON = REPO_ROOT / "data" / "intent_evaluation_cases.json"
MUSIC_INTENTS = sorted(VALID_INTENTS - {OUT_OF_SCOPE_INTENT})


@pytest.fixture(scope="module")
def trained_classifier() -> IntentClassifier:
    return IntentClassifier().train()


# --- Intent training data ---


def test_training_csv_loads():
    texts, intents = load_training_data(TRAINING_CSV)
    assert len(texts) == len(intents)
    assert len(texts) >= 150


def test_training_csv_has_required_columns():
    with TRAINING_CSV.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["text", "intent"]


def test_training_csv_only_uses_approved_labels():
    _, intents = load_training_data(TRAINING_CSV)
    assert set(intents) <= VALID_INTENTS


def test_training_csv_rejects_blank_examples(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("text,intent\n,workout\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_data(bad_csv)


def test_training_csv_rejects_unknown_intent_label(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("text,intent\nplay something,not_a_real_intent\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_data(bad_csv)


def test_training_csv_missing_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        load_training_data(REPO_ROOT / "data" / "does_not_exist.csv")


def test_training_data_has_enough_examples_per_class():
    _, intents = load_training_data(TRAINING_CSV)
    counts = Counter(intents)
    for intent in MUSIC_INTENTS:
        assert counts[intent] >= 20, f"{intent} has only {counts[intent]} examples"
    assert counts[OUT_OF_SCOPE_INTENT] >= 30


def test_evaluation_cases_are_not_exact_duplicates_of_training_rows():
    texts, _ = load_training_data(TRAINING_CSV)
    train_texts = {text.strip() for text in texts}
    with EVALUATION_JSON.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    overlap = [case["text"] for case in cases if case["text"].strip() and case["text"].strip() in train_texts]
    assert overlap == []


# --- Classifier training and prediction ---


def test_training_is_deterministic():
    text = "I need energetic music for a workout"
    first = IntentClassifier().train().predict(text)
    second = IntentClassifier().train().predict(text)
    assert first.intent == second.intent
    assert first.confidence == pytest.approx(second.confidence)
    assert first.margin == pytest.approx(second.margin)


@pytest.mark.parametrize(
    ("text", "expected_intent"),
    [
        ("I need energetic music for a workout", "workout"),
        ("Give me calm music while I study", "study"),
        ("Calm music for a quiet evening", "relax"),
        ("Dance music for a party tonight", "party"),
        ("Very calm music for falling asleep", "sleep"),
        ("I feel down and want something uplifting", "mood_boost"),
    ],
)
def test_valid_requests_accepted_with_expected_intent(trained_classifier, text, expected_intent):
    prediction = trained_classifier.predict(text)
    assert prediction.accepted is True
    assert prediction.intent == expected_intent
    assert prediction.reason == ""


def test_out_of_scope_request_is_rejected(trained_classifier):
    prediction = trained_classifier.predict("What is the weather tomorrow?")
    assert prediction.accepted is False
    assert prediction.reason != ""


def test_empty_input_is_rejected(trained_classifier):
    prediction = trained_classifier.predict("")
    assert prediction.accepted is False
    assert prediction.intent is None


def test_whitespace_only_input_is_rejected(trained_classifier):
    prediction = trained_classifier.predict("      ")
    assert prediction.accepted is False
    assert prediction.intent is None


def test_too_short_input_is_rejected(trained_classifier):
    prediction = trained_classifier.predict("hi")
    assert prediction.accepted is False
    assert prediction.intent is None


def test_too_long_input_is_rejected(trained_classifier):
    prediction = trained_classifier.predict("music " * 100)
    assert prediction.accepted is False
    assert prediction.intent is None


def test_non_string_input_raises_type_error(trained_classifier):
    with pytest.raises(TypeError):
        trained_classifier.predict(12345)


def test_predict_before_train_raises_runtime_error():
    with pytest.raises(RuntimeError):
        IntentClassifier().predict("I need energetic music for a workout")


@pytest.mark.parametrize(
    "text",
    ["I need energetic music for a workout", "What is the weather tomorrow?", "vibes"],
)
def test_confidence_and_margin_are_within_valid_ranges(trained_classifier, text):
    prediction = trained_classifier.predict(text)
    assert 0.0 <= prediction.confidence <= 1.0
    assert 0.0 <= prediction.second_confidence <= 1.0
    assert prediction.margin >= 0.0


def test_rejected_prediction_carries_guardrail_message(trained_classifier):
    prediction = trained_classifier.predict("What is the weather tomorrow?")
    assert prediction.accepted is False
    assert "music-listening intent" in prediction.reason


# --- Boundary / edge cases ---


def test_input_one_char_below_min_length_is_rejected(trained_classifier):
    text = "a" * (MIN_INPUT_LENGTH - 1)
    prediction = trained_classifier.predict(text)
    assert prediction.accepted is False
    assert prediction.intent is None


def test_input_exactly_at_min_length_reaches_model_inference(trained_classifier):
    text = "a" * MIN_INPUT_LENGTH
    prediction = trained_classifier.predict(text)
    assert prediction.normalized_text == text
    assert prediction.intent is not None


def test_input_exactly_at_max_length_reaches_model_inference(trained_classifier):
    text = ("a" * MAX_INPUT_LENGTH)
    assert len(text) == MAX_INPUT_LENGTH
    prediction = trained_classifier.predict(text)
    assert prediction.normalized_text == text
    assert prediction.intent is not None


def test_input_one_char_above_max_length_is_rejected(trained_classifier):
    text = "a" * (MAX_INPUT_LENGTH + 1)
    prediction = trained_classifier.predict(text)
    assert prediction.accepted is False
    assert prediction.intent is None
