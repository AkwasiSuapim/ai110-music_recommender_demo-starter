"""Automated evaluation harness for the specialized intent classifier.

Trains the TF-IDF + LogisticRegression classifier on the committed training
CSV, runs every case in data/intent_evaluation_cases.json through it, and
compares the result against the deterministic keyword baseline. All metrics
in the summary are computed from this run -- nothing is hard-coded.

Usage:
    python3 scripts/evaluate_intent_model.py
    python3 scripts/evaluate_intent_model.py --output evidence/evaluation_results.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, TextIO

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ai.intent_classifier import IntentClassifier, IntentPrediction, OUT_OF_SCOPE_INTENT  # noqa: E402
from src.ai.keyword_baseline import predict_intent as keyword_predict_intent  # noqa: E402

EVALUATION_CASES_PATH = REPO_ROOT / "data" / "intent_evaluation_cases.json"

INTENT_LABELS = ["workout", "study", "relax", "party", "sleep", "mood_boost", "out_of_scope"]
REJECTED_LABEL = "__rejected__"
CONFUSION_LABELS = INTENT_LABELS + [REJECTED_LABEL]
ACTION_LABELS = ["accept", "reject"]

REJECTED_LABEL_EXPLANATION = (
    "'__rejected__' marks requests the classifier rejected before committing to an "
    "intent (low confidence or an ambiguous margin between the top two candidates). "
    "It is kept separate from a genuine 'out_of_scope' prediction, which is the "
    "model actively and confidently classifying the request as out of scope."
)

ACCURACY_METRICS_EXPLANATION = (
    "Two different, both-correct 'intent accuracy' numbers appear below because they "
    "answer different questions:\n"
    "  Raw top-class intent accuracy -- numerator: cases where the model's raw top-1 "
    "predicted class (IntentPrediction.intent) equals expected_intent, PLUS every case "
    "whose expected_intent is null (there is no intent to compare, so it is counted as "
    "trivially correct); denominator: all evaluated cases (total_cases). This ignores "
    "whether the guardrail actually accepted the request -- a case can be 'raw-correct' "
    "even if it was rejected for low confidence or an ambiguous margin.\n"
    "  Guardrail-aware delivered intent accuracy -- numerator: cases where the label the "
    "system actually delivered (accept -> its intent; reject -> 'out_of_scope' only if "
    "that was the model's genuine top guess, otherwise '__rejected__') equals "
    "expected_intent; denominator: only the eligible cases (expected_intent is not null, "
    "intent_eligible_count). This is the same accounting as the classification report's "
    "weighted recall, since every eligible case falls into exactly one true class and "
    "'__rejected__'/mismatched predictions never count as a true positive for any class.\n"
    "The gap between the two numbers has two causes: (1) the raw metric's denominator "
    "includes 4 guardrail-only cases with no real expected_intent, which it counts as "
    "automatically correct; (2) among the 43 eligible cases, some requests whose raw top "
    "guess matched expected_intent were still rejected by the confidence/margin guardrail "
    "before that intent was ever delivered, so they count as correct for the raw metric "
    "but incorrect for the delivered metric."
)


def _confusion_prediction_label(prediction: IntentPrediction) -> str:
    """Map a raw IntentPrediction onto the confusion-matrix label set.

    Accepted predictions use their real intent. A prediction the model
    itself classified as out_of_scope is reported as out_of_scope (a
    genuine, confident classification). Every other rejection -- low
    confidence or ambiguous margin against a real intent, or no usable
    top guess at all -- is reported as REJECTED_LABEL rather than being
    folded into out_of_scope, since those rejections are not necessarily
    out-of-scope predictions.
    """
    if prediction.accepted:
        return prediction.intent
    if prediction.intent == OUT_OF_SCOPE_INTENT:
        return OUT_OF_SCOPE_INTENT
    return REJECTED_LABEL


def load_evaluation_cases(path: Path = EVALUATION_CASES_PATH) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation cases file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Evaluation cases file must contain a non-empty list: {path}")
    return cases


def _format_confusion_matrix(row_labels: List[str], col_labels: List[str], matrix: List[List[int]]) -> List[str]:
    """Render a confusion matrix as fixed-width text lines for the evidence report."""
    col_width = max(len(label) for label in [*row_labels, *col_labels, "true\\pred"]) + 2
    lines = ["true\\pred".ljust(col_width) + "".join(label.rjust(col_width) for label in col_labels)]
    for row_label, row in zip(row_labels, matrix):
        lines.append(row_label.ljust(col_width) + "".join(str(value).rjust(col_width) for value in row))
    return lines


def evaluate(cases: List[Dict], classifier: IntentClassifier, out: TextIO) -> Dict:
    """Run every case through the trained classifier and the keyword baseline."""
    intent_correct = 0
    action_correct = 0
    accepted_count = 0
    rejected_count = 0
    low_confidence_count = 0
    ambiguous_margin_count = 0
    trained_correct = 0
    baseline_correct = 0
    failed_cases: List[str] = []

    eligible_true: List[str] = []
    eligible_pred: List[str] = []
    action_true: List[str] = []
    action_pred: List[str] = []
    true_accept_count = 0
    false_accept_count = 0
    true_reject_count = 0
    false_reject_count = 0

    for index, case in enumerate(cases):
        text = case["text"]
        expected_intent = case.get("expected_intent")
        expected_action = case["expected_action"]
        description = case.get("description", "")

        prediction = classifier.predict(text)
        baseline = keyword_predict_intent(text)

        actual_action = "accept" if prediction.accepted else "reject"
        this_intent_correct = expected_intent is None or prediction.intent == expected_intent
        this_action_correct = actual_action == expected_action
        this_trained_correct = this_intent_correct and this_action_correct
        this_baseline_action = "accept" if baseline.accepted else "reject"
        this_baseline_correct = (
            (expected_intent is None or baseline.intent == expected_intent)
            and this_baseline_action == expected_action
        )

        if this_intent_correct:
            intent_correct += 1
        if this_action_correct:
            action_correct += 1
        if prediction.accepted:
            accepted_count += 1
        else:
            rejected_count += 1
        if prediction.intent is not None and not prediction.accepted and prediction.confidence < 0.35:
            low_confidence_count += 1
        if prediction.intent is not None and not prediction.accepted and prediction.margin < 0.05 and prediction.confidence >= 0.35:
            ambiguous_margin_count += 1
        if this_trained_correct:
            trained_correct += 1
        if this_baseline_correct:
            baseline_correct += 1
        if not this_trained_correct:
            failed_cases.append(f"[{index}] {text!r} -> {description}")

        action_true.append(expected_action)
        action_pred.append(actual_action)
        if expected_action == "accept" and actual_action == "accept":
            true_accept_count += 1
        elif expected_action == "accept" and actual_action == "reject":
            false_reject_count += 1
        elif expected_action == "reject" and actual_action == "accept":
            false_accept_count += 1
        elif expected_action == "reject" and actual_action == "reject":
            true_reject_count += 1

        if expected_intent is not None:
            eligible_true.append(expected_intent)
            eligible_pred.append(_confusion_prediction_label(prediction))

        result_line = (
            f"[{index}] expected_intent={expected_intent!r} expected_action={expected_action!r} "
            f"| predicted_intent={prediction.intent!r} confidence={prediction.confidence:.3f} "
            f"margin={prediction.margin:.3f} action={actual_action!r} "
            f"| baseline_intent={baseline.intent!r} baseline_action={this_baseline_action!r} "
            f"| intent_correct={this_intent_correct} action_correct={this_action_correct} "
            f"| description={description}"
        )
        print(result_line, file=out)

    total = len(cases)
    intent_accuracy = intent_correct / total * 100
    action_accuracy = action_correct / total * 100
    trained_accuracy = trained_correct / total * 100
    baseline_accuracy = baseline_correct / total * 100

    summary = {
        "total_cases": total,
        "intent_correct": intent_correct,
        "intent_accuracy_pct": round(intent_accuracy, 2),
        "action_correct": action_correct,
        "action_accuracy_pct": round(action_accuracy, 2),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "low_confidence_count": low_confidence_count,
        "ambiguous_margin_count": ambiguous_margin_count,
        "trained_accuracy_pct": round(trained_accuracy, 2),
        "baseline_accuracy_pct": round(baseline_accuracy, 2),
        "accuracy_difference_pct": round(trained_accuracy - baseline_accuracy, 2),
        "failed_cases": failed_cases,
    }

    # --- Precision / recall / F1 metrics (scikit-learn), computed from this run only ---

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        eligible_true, eligible_pred, labels=INTENT_LABELS, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        eligible_true, eligible_pred, labels=INTENT_LABELS, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        eligible_true, eligible_pred, labels=INTENT_LABELS, average="weighted", zero_division=0
    )
    intent_confusion_full = confusion_matrix(eligible_true, eligible_pred, labels=CONFUSION_LABELS)
    intent_confusion_rows = intent_confusion_full[: len(INTENT_LABELS)]
    intent_classification_report_text = classification_report(
        eligible_true,
        eligible_pred,
        labels=INTENT_LABELS,
        target_names=INTENT_LABELS,
        zero_division=0,
        digits=4,
    )

    action_precision, action_recall, action_f1, action_support = precision_recall_fscore_support(
        action_true, action_pred, labels=ACTION_LABELS, zero_division=0
    )
    action_macro_precision, action_macro_recall, action_macro_f1, _ = precision_recall_fscore_support(
        action_true, action_pred, labels=ACTION_LABELS, average="macro", zero_division=0
    )
    action_confusion = confusion_matrix(action_true, action_pred, labels=ACTION_LABELS)

    intent_metrics = {
        label: {
            "precision": round(float(per_class_precision[i]), 4),
            "recall": round(float(per_class_recall[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(per_class_support[i]),
        }
        for i, label in enumerate(INTENT_LABELS)
    }
    action_metrics = {
        label: {
            "precision": round(float(action_precision[i]), 4),
            "recall": round(float(action_recall[i]), 4),
            "f1": round(float(action_f1[i]), 4),
            "support": int(action_support[i]),
        }
        for i, label in enumerate(ACTION_LABELS)
    }

    guardrail_aware_intent_correct = sum(1 for true_label, pred_label in zip(eligible_true, eligible_pred) if true_label == pred_label)
    guardrail_aware_intent_accuracy = (
        guardrail_aware_intent_correct / len(eligible_true) * 100 if eligible_true else 0.0
    )

    summary.update(
        {
            # Aliases for `intent_correct` / `intent_accuracy_pct` above, named to make
            # clear exactly what that pre-existing metric measures (see
            # ACCURACY_METRICS_EXPLANATION): the model's raw top-1 predicted class vs.
            # expected_intent, over ALL cases, ignoring guardrail accept/reject entirely.
            "raw_top_class_intent_correct": intent_correct,
            "raw_top_class_intent_accuracy_pct": round(intent_accuracy, 2),
            # Numerator: eligible cases where the label the system actually delivered
            # (accept -> its intent; reject -> out_of_scope only if that was the genuine
            # top guess, else __rejected__) equals expected_intent.
            # Denominator: intent_eligible_count (expected_intent is not null).
            # Equivalent to the classification report's weighted recall.
            "guardrail_aware_intent_correct": guardrail_aware_intent_correct,
            "guardrail_aware_intent_accuracy_pct": round(guardrail_aware_intent_accuracy, 2),
            "intent_eligible_count": len(eligible_true),
            "intent_metrics": intent_metrics,
            "intent_macro": {
                "precision": round(float(macro_precision), 4),
                "recall": round(float(macro_recall), 4),
                "f1": round(float(macro_f1), 4),
            },
            "intent_weighted": {
                "precision": round(float(weighted_precision), 4),
                "recall": round(float(weighted_recall), 4),
                "f1": round(float(weighted_f1), 4),
            },
            "intent_confusion_matrix": {
                "true_labels": INTENT_LABELS,
                "predicted_labels": CONFUSION_LABELS,
                "matrix": intent_confusion_rows.tolist(),
            },
            "rejected_label_explanation": REJECTED_LABEL_EXPLANATION,
            "action_metrics": action_metrics,
            "action_macro": {
                "precision": round(float(action_macro_precision), 4),
                "recall": round(float(action_macro_recall), 4),
                "f1": round(float(action_macro_f1), 4),
            },
            "action_confusion_matrix": {
                "true_labels": ACTION_LABELS,
                "predicted_labels": ACTION_LABELS,
                "matrix": action_confusion.tolist(),
            },
            "true_accept_count": true_accept_count,
            "false_accept_count": false_accept_count,
            "true_reject_count": true_reject_count,
            "false_reject_count": false_reject_count,
        }
    )

    print("\n=== Evaluation Summary ===", file=out)
    print(f"Total cases: {summary['total_cases']}", file=out)
    print(
        f"Raw top-class intent accuracy: {summary['raw_top_class_intent_correct']}/{summary['total_cases']} "
        f"({summary['raw_top_class_intent_accuracy_pct']:.2f}%)  "
        f"[formerly labeled 'Intent classification correct']",
        file=out,
    )
    print(
        f"Guardrail-aware delivered intent accuracy: {summary['guardrail_aware_intent_correct']}/"
        f"{summary['intent_eligible_count']} ({summary['guardrail_aware_intent_accuracy_pct']:.2f}%)  "
        f"[matches the classification report's weighted recall below]",
        file=out,
    )
    print(f"\n{ACCURACY_METRICS_EXPLANATION}\n", file=out)
    print(
        f"Guardrail action correct: {summary['action_correct']}/{summary['total_cases']} "
        f"({summary['action_accuracy_pct']:.2f}%)",
        file=out,
    )
    print(f"Accepted cases: {summary['accepted_count']}", file=out)
    print(f"Rejected cases: {summary['rejected_count']}", file=out)
    print(f"Low-confidence rejections: {summary['low_confidence_count']}", file=out)
    print(f"Ambiguous-margin rejections: {summary['ambiguous_margin_count']}", file=out)
    print(f"Trained classifier accuracy: {summary['trained_accuracy_pct']:.2f}%", file=out)
    print(f"Keyword baseline accuracy: {summary['baseline_accuracy_pct']:.2f}%", file=out)
    print(
        f"Difference (trained - baseline): {summary['accuracy_difference_pct']:.2f} points",
        file=out,
    )
    if summary["failed_cases"]:
        print(f"Failed cases ({len(summary['failed_cases'])}):", file=out)
        for failure in summary["failed_cases"]:
            print(f"  - {failure}", file=out)
    else:
        print("Failed cases: none", file=out)

    print("\n=== Intent Classification Metrics (scikit-learn) ===", file=out)
    print(f"Eligible cases (expected_intent is not null): {summary['intent_eligible_count']}", file=out)
    print(REJECTED_LABEL_EXPLANATION, file=out)
    print("\nPer-intent precision / recall / F1 / support:", file=out)
    for label in INTENT_LABELS:
        metrics = summary["intent_metrics"][label]
        print(
            f"  {label:<13} precision={metrics['precision'] * 100:6.2f}%  "
            f"recall={metrics['recall'] * 100:6.2f}%  "
            f"f1={metrics['f1'] * 100:6.2f}%  "
            f"support={metrics['support']}",
            file=out,
        )
    macro = summary["intent_macro"]
    weighted = summary["intent_weighted"]
    print(
        f"\nMacro-average:    precision={macro['precision'] * 100:.2f}%  "
        f"recall={macro['recall'] * 100:.2f}%  f1={macro['f1'] * 100:.2f}%",
        file=out,
    )
    print(
        f"Weighted-average: precision={weighted['precision'] * 100:.2f}%  "
        f"recall={weighted['recall'] * 100:.2f}%  f1={weighted['f1'] * 100:.2f}%",
        file=out,
    )

    print("\nIntent confusion matrix (rows = expected intent, columns = predicted label):", file=out)
    for line in _format_confusion_matrix(
        INTENT_LABELS, CONFUSION_LABELS, summary["intent_confusion_matrix"]["matrix"]
    ):
        print(f"  {line}", file=out)

    print("\nscikit-learn classification_report:", file=out)
    print(intent_classification_report_text, file=out)

    print("=== Guardrail Action Metrics (accept vs reject, all cases) ===", file=out)
    for label in ACTION_LABELS:
        metrics = summary["action_metrics"][label]
        print(
            f"  {label:<8} precision={metrics['precision'] * 100:6.2f}%  "
            f"recall={metrics['recall'] * 100:6.2f}%  "
            f"f1={metrics['f1'] * 100:6.2f}%  "
            f"support={metrics['support']}",
            file=out,
        )
    action_macro = summary["action_macro"]
    print(
        f"\nMacro-average action: precision={action_macro['precision'] * 100:.2f}%  "
        f"recall={action_macro['recall'] * 100:.2f}%  f1={action_macro['f1'] * 100:.2f}%",
        file=out,
    )

    print(
        f"\nTrue accept  (expected accept, actually accepted): {summary['true_accept_count']}",
        file=out,
    )
    print(
        f"False reject (expected accept, actually rejected): {summary['false_reject_count']}",
        file=out,
    )
    print(
        f"False accept (expected reject, actually accepted): {summary['false_accept_count']}",
        file=out,
    )
    print(
        f"True reject  (expected reject, actually rejected): {summary['true_reject_count']}",
        file=out,
    )

    print("\nAction confusion matrix (rows = expected action, columns = actual action):", file=out)
    for line in _format_confusion_matrix(ACTION_LABELS, ACTION_LABELS, summary["action_confusion_matrix"]["matrix"]):
        print(f"  {line}", file=out)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the specialized intent classifier.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to also write the full evaluation report to a file.",
    )
    args = parser.parse_args(argv)

    cases = load_evaluation_cases()
    classifier = IntentClassifier().train()

    outputs: List[TextIO] = [sys.stdout]
    output_file = None
    if args.output:
        output_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8")
        outputs.append(output_file)

    class _Tee:
        def write(self, text: str) -> None:
            for stream in outputs:
                stream.write(text)

        def flush(self) -> None:
            for stream in outputs:
                stream.flush()

    evaluate(cases, classifier, _Tee())

    if output_file is not None:
        output_file.close()
        print(f"\nFull report written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
