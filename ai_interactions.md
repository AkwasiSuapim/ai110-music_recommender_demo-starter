# AI Interaction Log

This is a concise development record of observable tasks, decisions, outputs, and verification. It does not reproduce private chain-of-thought or claim unsupported verbatim prompt history.

## Project Planning

Francis decided to extend the CodePath Module 3 Music Recommender Simulation with a natural-language intent classifier. The agreed system boundary was: interpret a short request with AI, map an accepted intent into the structured preference format already used by the project, and preserve the original deterministic song-ranking logic.

The Project 4 scope was kept to a local CLI and offline model. Broader product ideas such as web services, accounts, external music APIs, databases, and the longer-term Intentune platform were not included in this deadline-focused submission.

## Coding Agent Workflow

The coding agent was tasked with implementing and verifying the applied-AI extension without replacing the original scoring system. Its completed work is observable in the repository:

**Created:**

- `src/ai/__init__.py`
- `src/ai/intent_classifier.py`
- `src/ai/intent_profiles.py`
- `src/ai/keyword_baseline.py`
- `data/intent_training_data.csv`
- `data/intent_evaluation_cases.json`
- `scripts/evaluate_intent_model.py`
- `scripts/run_sample_cases.py`
- `tests/test_intent_classifier.py`
- `tests/test_intent_profiles.py`
- `tests/test_intent_evaluation.py`
- `tests/test_ai_recommendation_integration.py`
- generated files under `evidence/`

**Modified:**

- `src/main.py` to add natural-language CLI flow and guarded integration
- `requirements.txt` to include scikit-learn and pytest
- `.gitignore` as part of repository housekeeping

The agent implemented TF-IDF plus Logistic Regression classification, catalog-aware intent profiles, CLI validation, confidence and ambiguity guardrails, a keyword baseline, a held-out evaluation harness, evidence generation, and automated tests. Francis’s owner decisions included retaining deterministic recommendation scoring, limiting supported intents, rejecting unsupported inputs before scoring, and excluding unimplemented product features. Verification used `python -m pytest -q`, four direct CLI requests, generated sample runs, and `python scripts/evaluate_intent_model.py --output evidence/evaluation_results.txt` from an activated virtual environment.

## Guardrail Design

The classifier rejects blank input, stripped input shorter than 3 characters, input longer than 500 characters, a top `out_of_scope` prediction, music-intent confidence below `0.35`, or a top-two probability margin below `0.05`. These checks were chosen to avoid converting unsupported or ambiguous language into made-up preferences. The CLI exits on rejection before loading the catalog or calling recommendation scoring, and integration tests explicitly verify that boundary.

## Helpful Suggestion

AI recommended the modular classifier-to-profile-to-recommender structure. This was helpful because natural-language interpretation could be evaluated and guarded independently while the established weighted song ranking stayed deterministic and explainable. Francis verified the design with component tests, tests comparing different profiles and recommendation outcomes, and end-to-end runs showing the complete accepted path.

## Flawed or Incomplete Suggestion

During an earlier missing-data discussion, AI suggested excluding missing song features from the score denominator. Francis identified that this could unfairly boost songs with unknown metadata when a user explicitly cared about the missing feature—for example, a high-energy requirement. Because the current 20-song CSV has complete required scoring fields and missing-feature behavior was outside the submission scope, that suggestion was rejected rather than implemented. The issue remains a longer-term design consideration, not a claimed feature of this project.

## Model Calibration Issue

Initial reasoning assumed the default Logistic Regression setting `C=1.0` would work with the planned confidence threshold. Actual probability checks showed near-uniform class probabilities on this small, seven-label dataset, causing clear in-vocabulary requests to be rejected below the threshold. The implementation changed to `C=10.0`, then verified the choice with deterministic classifier tests, six representative intent tests, end-to-end samples, and the held-out evaluation script. This is retained as a dataset-specific calibration choice, not a generally optimal setting; figurative paraphrases still fail in some cases.

## Evaluation Comparison

The evaluation harness computes results from 47 committed held-out cases and compares the trained model with a deterministic keyword baseline used only for evaluation.

| System | Accuracy |
| --- | ---: |
| Trained TF-IDF classifier | 80.85% |
| Keyword baseline | 70.21% |
| Difference | +10.64 percentage points |

The initial evaluation emphasized accuracy. Francis requested precision, recall, F1, and confusion matrices because accuracy alone did not explain whether the guardrails were allowing unsupported requests or rejecting valid ones. AI assisted by proposing this metric expansion, but every reported value was computed by the project's evaluation script and verified through automated tests.

The expanded evaluation distinguishes raw top-class intent accuracy (41/47, 87.23%) from guardrail-aware delivered intent accuracy (34/43, 79.07%) and action accuracy (39/47, 82.98%). It revealed zero false accepts and eight false rejects. The main weakness was low recall for indirect `relax` requests: relax recall was 40.00%, with figurative and overlapping calm/study language causing failures. This led to a more honest interpretation of the model as conservative rather than uniformly accurate. No retraining, model setting, guardrail threshold, or application behavior changed during this documentation update.

## Human Verification

Francis reviewed the classifier output, mapped profiles, recommendation rankings, explanations, guardrail messages, confusion matrices, and case-level evaluation failures. He required reproducible evidence rather than accepting generated claims: the complete test suite contains 107 passing tests; three successful CLI samples produced confidences of `0.790`, `0.795`, and `0.803`; unrelated and whitespace-only requests exited safely without recommendations; and the evaluation report was regenerated from current code. The added precision, recall, and F1 tests verify the calculations. Documentation reports both strengths and failures so the submission reflects observed behavior rather than an AI-generated success narrative.
