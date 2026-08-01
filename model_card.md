# Model Card: Intentune AI Intent Classifier

## Model Overview

Intentune uses a scikit-learn pipeline with TF-IDF unigram and bigram features and Logistic Regression. It predicts one of seven labels: six supported listening intents plus `out_of_scope`. The classifier’s narrow role is to interpret a short natural-language request. After guardrails accept a prediction, separate catalog-aware mapping creates structured preferences and the existing deterministic recommender scores and ranks songs.

## Intended Use

The classifier is intended to interpret short English music requests entered through the Intentune CLI. It supports requests for workout, study, relaxation, party, sleep, and mood-boost listening contexts, and rejects requests outside that narrow scope.

## Out-of-Scope Uses

The model is not intended for:

- general-purpose question answering;
- mental-health or emotional-state diagnosis;
- safety-critical decisions;
- professional or authoritative music classification; or
- claims about the objective quality of a song, artist, genre, or listener preference.

## Training Data

The committed training CSV contains 164 examples:

| Label | Examples |
| --- | ---: |
| `workout` | 22 |
| `study` | 22 |
| `relax` | 22 |
| `party` | 22 |
| `sleep` | 22 |
| `mood_boost` | 22 |
| `out_of_scope` | 32 |

The examples are synthetic/manual, project-specific request phrasings. They contain no personal data and no copyrighted song lyrics.

## Evaluation Data

The evaluation file contains 47 held-out cases: unseen requests for the six music intents, unrelated `out_of_scope` questions, and explicit guardrail cases for empty, whitespace-only, overlong, and ambiguous input. An automated test verifies that no nonblank evaluation request is an exact duplicate of a training row.

## Features and Labels

TF-IDF features are built from lowercased word unigrams and bigrams with sublinear term frequency. The prediction labels are:

- `workout`
- `study`
- `relax`
- `party`
- `sleep`
- `mood_boost`
- `out_of_scope`

## Algorithmic Approach in Plain Language

TF-IDF turns the words and adjacent word pairs in a request into numbers. It gives more weight to language that is informative for a request and less weight to language common across many examples. Logistic Regression learns how those numerical patterns relate to each intent, then estimates a probability for every label. Intentune examines the highest probability and its separation from the runner-up before deciding whether to trust the prediction.

## Guardrails

- Input must be a string; non-string values raise a clear type error at the classifier boundary.
- Blank or whitespace-only input is rejected before inference.
- Stripped input shorter than 3 characters is rejected.
- Stripped input longer than 500 characters is rejected.
- A top prediction of `out_of_scope` is always rejected.
- A music-intent prediction below `0.35` confidence is rejected.
- A top-two probability margin below `0.05` is rejected as ambiguous.
- After any rejection, the CLI returns a user-facing guidance message and does not call the recommendation engine.

These checks reduce unsupported recommendations, but they do not guarantee that every accepted prediction is correct.

## Evaluation Results

The reproducible evaluation contains 47 total cases. Forty-three cases have a defined expected intent and are eligible for the guardrail-aware classification report.

| Measure | Result |
| --- | ---: |
| Trained classifier accuracy | 80.85% |
| Keyword baseline accuracy | 70.21% |
| Improvement | +10.64 percentage points |
| Raw top-class intent accuracy | 41/47 (87.23%) |
| Guardrail-aware delivered intent accuracy | 34/43 (79.07%) |
| Guardrail action correct | 39/47 (82.98%) |
| Accepted / rejected | 22 / 25 |
| Low-confidence rejections | 9 |
| Ambiguous-margin rejections | 0 |
| Macro precision / recall / F1 | 95.41% / 74.29% / 81.84% |
| Weighted precision / recall / F1 | 94.93% / 79.07% / 84.53% |

Raw top-class accuracy examines the model's highest-probability class before guardrails and includes all 47 cases, including four guardrail-only cases without an expected intent. Guardrail-aware delivered accuracy evaluates what the system actually returns, represents low-confidence or ambiguous decisions as `__rejected__`, and uses only the 43 intent-eligible cases. The 79.07% delivered metric is therefore the more useful measure of end-to-end intent behavior; it is not interchangeable with the 87.23% raw result or the 80.85% classifier/baseline comparison score.

### Per-intent results

| Intent | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `workout` | 100.00% | 80.00% | 88.89% | 5 |
| `study` | 75.00% | 60.00% | 66.67% | 5 |
| `relax` | 100.00% | 40.00% | 57.14% | 5 |
| `party` | 100.00% | 100.00% | 100.00% | 5 |
| `sleep` | 100.00% | 80.00% | 88.89% | 5 |
| `mood_boost` | 100.00% | 60.00% | 75.00% | 5 |
| `out_of_scope` | 92.86% | 100.00% | 96.30% | 13 |

### Guardrail action results

| Action | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `accept` | 100.00% | 73.33% | 84.62% | 30 |
| `reject` | 68.00% | 100.00% | 80.95% | 17 |

The action confusion matrix contains zero false accepts and eight false rejects. This is conservative guardrail behavior: accepted requests are precise, but recall is weaker because some valid requests are rejected. `relax` is the primary intent weakness at 40.00% recall. Study and relax can overlap because both may contain calm, quiet, background, or low-energy language, while figurative relax and mood-boost language is harder than direct requests. Examples include “in the zone,” porch-reading language, and “melt into the sofa.” Because these percentages come from only 47 evaluation cases, each individual example has a noticeable effect on the result.

Blank, whitespace-only, overlong, ambiguous, and clear non-music cases were rejected in the evaluation set. The CLI tests verify that rejected input never reaches recommendation scoring. Successful end-to-end examples were accepted as workout (`0.790` confidence), study (`0.795`), and mood boost (`0.803`).

## Limitations and Biases

- The model focuses on English-language requests.
- Synthetic/manual examples can encode the vocabulary and assumptions of their authors.
- The training set is small.
- The six activity categories were chosen by developers and do not cover all listening contexts.
- Wording, music-intent definitions, and the catalog may contain cultural, genre, and mood biases.
- Figurative, indirect, slang-heavy, or unfamiliar language remains difficult.
- Probability confidence is only imperfectly calibrated and is not certainty.
- The `C=10.0` and threshold choices are calibrated to this dataset.
- The 20-song catalog limits the range and diversity of recommendations.
- Deterministic scoring uses exact genre and mood matching, so related terms receive no partial credit.
- Profile mapping and recommendations are constrained by metadata actually available in the catalog.

## Potential Misuse

Misuse includes treating confidence as certainty, using inferred `mood_boost` intent as an emotional or psychological diagnosis, claiming rankings measure objective musical quality, or expanding request logging in a privacy-invasive way. The classifier should not be repurposed beyond its documented music-request scope without new data, evaluation, and safeguards.

## Misuse Prevention

Intentune provides a clear rejection message for unsupported or uncertain input and states a narrow intended use. Application logs include request length and prediction metadata rather than the full raw request. A rejected prediction cannot create a profile or call the recommender. For accepted requests, deterministic score contributions make the ranking inspectable instead of presenting an unexplained AI judgment.

## Reliability Findings

The surprising calibration finding was that default Logistic Regression regularization (`C=1.0`) produced probabilities that were too close together for the selected confidence threshold on this small seven-class dataset. Apparently clear requests were therefore rejected. Changing to `C=10.0` made the verified clear requests pass while uncertain examples remained subject to the same confidence and margin checks. The current metrics show that the model performs well when it accepts a request—accept precision is 100.00%—but needs better linguistic coverage to improve its 73.33% accept recall. Indirect and figurative paraphrases, especially relaxation requests, remain difficult, so this dataset-specific calibration improved measured behavior without solving language understanding or providing universal reliability.

## AI Collaboration Reflection

### How AI was used

AI assisted me with architecture planning, implementation suggestions, debugging, test planning, edge-case review, and documentation organization. I directed the scope, made product and reliability decisions, reviewed generated work, and verified the finished behavior. AI did not independently create or validate the entire project.

### Helpful AI suggestion

A particularly helpful suggestion was to separate natural-language intent classification from deterministic song scoring. That modular boundary made the AI feature substantial and integrated while preserving the original recommender’s explainable ranking logic. I verified the boundary through unit tests for each component, integration tests showing that accepted requests reach scoring and rejected requests do not, and end-to-end CLI runs that display the predicted intent, structured profile, ranked songs, scores, and explanations.

### Flawed or incorrect AI suggestion

An earlier AI suggestion proposed excluding missing song features from the score denominator. I recognized that this could make a song with unknown metadata appear artificially strong when the listener explicitly required a feature such as high energy: less evidence would be scored as though it were equally complete evidence. Missing-feature scoring was outside this deadline-focused submission because the committed catalog is complete, but I rejected that shortcut and retained the concern for longer-term design rather than presenting it as solved here.

### What I accepted, rejected, and verified

I accepted the modular classifier-to-profile-to-recommender architecture, explicit confidence/out-of-scope guardrails, and a reproducible evaluation harness with a keyword baseline. I rejected unsupported deadline-expanding product features and the unsafe missing-feature scoring shortcut. I verified AI-assisted suggestions by running the complete suite of 107 passing tests, reviewing generated sample evidence, executing successful and rejected CLI cases, and regenerating the 47-case evaluation report instead of accepting unverified performance claims.

### System limitations and future improvements

The present system is constrained by a small synthetic/manual training set, six music intents, English wording, imperfect probability calibration, exact-match ranking features, and a 20-song catalog. Realistic next steps include larger and more representative data, broader intents, formal calibration analysis, a richer catalog, user feedback, optional provider integration, and stronger language representations. Each change would require fresh guardrail and held-out evaluation rather than assuming the current metrics transfer.

## Ethical Considerations

Privacy is supported by offline inference, no external API calls, no user tracking, and omission of full raw request text from application logs. Transparency comes from narrow intent labels, explicit rejection behavior, and deterministic recommendation explanations. Confidence must be communicated as a fallible probability rather than certainty. Both the authored training phrases and small catalog can encode cultural and genre biases, so the tool should remain a low-stakes music demo with a narrow intended use, not a system for judging people, emotions, artists, or musical quality.
