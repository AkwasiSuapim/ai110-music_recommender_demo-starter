import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_intent_model import (  # noqa: E402
    ACTION_LABELS,
    CONFUSION_LABELS,
    EVALUATION_CASES_PATH,
    INTENT_LABELS,
    REJECTED_LABEL,
    evaluate,
    load_evaluation_cases,
    main as evaluate_main,
)
from src.ai.intent_classifier import IntentClassifier  # noqa: E402

REQUIRED_SUMMARY_KEYS = {
    "total_cases",
    "intent_correct",
    "intent_accuracy_pct",
    "action_correct",
    "action_accuracy_pct",
    "accepted_count",
    "rejected_count",
    "low_confidence_count",
    "ambiguous_margin_count",
    "trained_accuracy_pct",
    "baseline_accuracy_pct",
    "accuracy_difference_pct",
    "failed_cases",
    "intent_eligible_count",
    "intent_metrics",
    "intent_macro",
    "intent_weighted",
    "intent_confusion_matrix",
    "action_metrics",
    "action_macro",
    "action_confusion_matrix",
    "true_accept_count",
    "false_accept_count",
    "true_reject_count",
    "false_reject_count",
    "raw_top_class_intent_correct",
    "raw_top_class_intent_accuracy_pct",
    "guardrail_aware_intent_correct",
    "guardrail_aware_intent_accuracy_pct",
}

PER_CLASS_METRIC_KEYS = {"precision", "recall", "f1", "support"}


def test_evaluation_script_loads_all_cases():
    cases = load_evaluation_cases()
    with EVALUATION_CASES_PATH.open(encoding="utf-8") as handle:
        raw_cases = json.load(handle)

    assert len(cases) == len(raw_cases)
    assert len(cases) >= 45


def test_result_summary_contains_required_metrics():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    assert REQUIRED_SUMMARY_KEYS <= summary.keys()


def test_computed_counts_equal_number_of_cases():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    assert summary["total_cases"] == len(cases)
    assert summary["accepted_count"] + summary["rejected_count"] == len(cases)
    assert summary["intent_correct"] <= summary["total_cases"]
    assert summary["action_correct"] <= summary["total_cases"]


def test_baseline_comparison_runs_and_produces_bounded_accuracy():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    assert 0.0 <= summary["baseline_accuracy_pct"] <= 100.0
    assert 0.0 <= summary["trained_accuracy_pct"] <= 100.0


def test_optional_output_file_is_created(tmp_path):
    output_path = REPO_ROOT / "evidence" / "_test_evaluation_output.txt"
    relative_output = "evidence/_test_evaluation_output.txt"
    try:
        exit_code = evaluate_main(["--output", relative_output])
        assert exit_code == 0
        assert output_path.is_file()
        content = output_path.read_text(encoding="utf-8")
        assert "Evaluation Summary" in content
    finally:
        if output_path.exists():
            output_path.unlink()


def test_evaluation_runs_successfully_even_with_imperfect_accuracy():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    assert summary["trained_accuracy_pct"] < 100.0
    assert summary["trained_accuracy_pct"] > 0.0


# --- Precision / recall / F1 metrics ---


def test_per_class_intent_metrics_are_present_for_every_fixed_label():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    assert set(summary["intent_metrics"].keys()) == set(INTENT_LABELS)
    for label in INTENT_LABELS:
        assert PER_CLASS_METRIC_KEYS <= summary["intent_metrics"][label].keys()


def test_macro_and_weighted_intent_metrics_are_present():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    assert {"precision", "recall", "f1"} <= summary["intent_macro"].keys()
    assert {"precision", "recall", "f1"} <= summary["intent_weighted"].keys()


def test_guardrail_accept_reject_metrics_are_present():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    assert set(summary["action_metrics"].keys()) == set(ACTION_LABELS)
    for label in ACTION_LABELS:
        assert PER_CLASS_METRIC_KEYS <= summary["action_metrics"][label].keys()
    assert {"precision", "recall", "f1"} <= summary["action_macro"].keys()


def test_confusion_matrices_are_present_and_shaped_correctly():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    intent_cm = summary["intent_confusion_matrix"]
    assert intent_cm["true_labels"] == INTENT_LABELS
    assert intent_cm["predicted_labels"] == CONFUSION_LABELS
    assert REJECTED_LABEL in intent_cm["predicted_labels"]
    assert len(intent_cm["matrix"]) == len(INTENT_LABELS)
    assert all(len(row) == len(CONFUSION_LABELS) for row in intent_cm["matrix"])

    action_cm = summary["action_confusion_matrix"]
    assert action_cm["true_labels"] == ACTION_LABELS
    assert action_cm["predicted_labels"] == ACTION_LABELS
    assert len(action_cm["matrix"]) == len(ACTION_LABELS)
    assert all(len(row) == len(ACTION_LABELS) for row in action_cm["matrix"])


def test_all_metric_values_are_fractions_between_zero_and_one():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    for label in INTENT_LABELS:
        metrics = summary["intent_metrics"][label]
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    for aggregate in (summary["intent_macro"], summary["intent_weighted"], summary["action_macro"]):
        assert 0.0 <= aggregate["precision"] <= 1.0
        assert 0.0 <= aggregate["recall"] <= 1.0
        assert 0.0 <= aggregate["f1"] <= 1.0

    for label in ACTION_LABELS:
        metrics = summary["action_metrics"][label]
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


def test_intent_supports_sum_to_eligible_case_count():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    total_support = sum(summary["intent_metrics"][label]["support"] for label in INTENT_LABELS)
    assert total_support == summary["intent_eligible_count"]
    assert summary["intent_eligible_count"] == sum(1 for case in cases if case.get("expected_intent") is not None)


def test_action_supports_sum_to_all_cases():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    total_support = sum(summary["action_metrics"][label]["support"] for label in ACTION_LABELS)
    assert total_support == summary["total_cases"] == len(cases)


def test_true_false_accept_reject_counts_add_up_to_total_cases():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    total = (
        summary["true_accept_count"]
        + summary["false_accept_count"]
        + summary["true_reject_count"]
        + summary["false_reject_count"]
    )
    assert total == summary["total_cases"]


def test_output_file_contains_new_precision_recall_metrics():
    output_path = REPO_ROOT / "evidence" / "_test_evaluation_metrics_output.txt"
    relative_output = "evidence/_test_evaluation_metrics_output.txt"
    try:
        exit_code = evaluate_main(["--output", relative_output])
        assert exit_code == 0
        content = output_path.read_text(encoding="utf-8")
        assert "Intent Classification Metrics" in content
        assert "Guardrail Action Metrics" in content
        assert "Macro-average" in content
        assert "Weighted-average" in content
        assert "confusion matrix" in content
        assert REJECTED_LABEL in content
        assert "Raw top-class intent accuracy" in content
        assert "Guardrail-aware delivered intent accuracy" in content
        assert "weighted recall" in content
    finally:
        if output_path.exists():
            output_path.unlink()


def test_raw_and_guardrail_aware_intent_accuracy_are_distinct_and_well_defined():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    # Raw metric: denominator is every case (including guardrail-only cases with no
    # real expected_intent, which are counted as trivially correct).
    assert summary["raw_top_class_intent_correct"] == summary["intent_correct"]
    assert summary["raw_top_class_intent_accuracy_pct"] == summary["intent_accuracy_pct"]
    assert summary["raw_top_class_intent_correct"] <= summary["total_cases"]

    # Guardrail-aware metric: denominator is only the eligible (real expected_intent) cases.
    assert summary["guardrail_aware_intent_correct"] <= summary["intent_eligible_count"]
    expected_pct = round(summary["guardrail_aware_intent_correct"] / summary["intent_eligible_count"] * 100, 2)
    assert summary["guardrail_aware_intent_accuracy_pct"] == expected_pct

    # On this dataset the two numbers are known to differ (raw counts some rejected-but
    # raw-correct guesses and all null-expected_intent cases as correct; guardrail-aware
    # does not) -- assert that difference is real, not asserted away.
    assert summary["raw_top_class_intent_accuracy_pct"] != summary["guardrail_aware_intent_accuracy_pct"]


def test_guardrail_aware_intent_accuracy_matches_classification_report_weighted_recall():
    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    summary = evaluate(cases, classifier, io.StringIO())

    delivered_fraction = summary["guardrail_aware_intent_correct"] / summary["intent_eligible_count"]
    # intent_weighted["recall"] is stored rounded to 4 decimal places; allow for that rounding.
    assert delivered_fraction == pytest.approx(summary["intent_weighted"]["recall"], abs=5e-5)


def test_zero_division_handling_for_a_class_with_no_eligible_cases():
    classifier = IntentClassifier().train()
    cases = [
        {
            "text": "What is the weather tomorrow?",
            "expected_intent": "out_of_scope",
            "expected_action": "reject",
            "description": "single out-of-scope case, no other intents present",
        }
    ]

    summary = evaluate(cases, classifier, io.StringIO())

    for label in INTENT_LABELS:
        if label == "out_of_scope":
            continue
        metrics = summary["intent_metrics"][label]
        assert metrics["support"] == 0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
