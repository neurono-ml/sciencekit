# Proposal

## Why

`metrics-foundation` shipped accuracy, precision/recall/F1, the confusion matrix and the
regression errors, but the classification suite still lacks the score-based metrics every
ranking and calibration workflow needs: ROC/AUC, precision-recall, log loss, balanced
accuracy and Matthews correlation. `SKGridSearchCV` must be able to select models by these
metrics, so they need free functions, scorer adapters and a probability-based scoring
contract.

## What Changes

- Add a `SKProbabilityScorer<F, M: SKClassifierPredictor<F>>` contract to
  `scoring-contracts`: a pure form over stored probabilities and a convenient form that
  calls `predict_probabilities`.
- Add binary and multiclass (one-vs-rest) ROC curve and ROC-AUC, with macro,
  micro and weighted averaging.
- Add precision-recall curve and average precision.
- Add binary/multiclass log loss, balanced accuracy and Matthews correlation coefficient.
- Add scorer adapters for AUC, average precision, log loss, balanced accuracy and MCC.
- Add an exact mergeable `SKAucAccumulator` (score/label pairs) with the documented O(n)
  memory profile; a sketch-based variant is a future accelerator.
- Validation: empty input, single-class AUC, invalid score range, length mismatch and
  non-finite values.
- Explicit non-goals: calibration curves, Brier score, ranking metrics beyond AUC/AP,
  `memmap2` and export.
- Companion tests: known fixtures, multiclass averaging, f32/f64 precision, large data,
  concurrency, accumulator merge and scorer equivalence.

No existing metric behavior changes and no breaking changes.

## Capabilities

### New Capabilities

- `ranking-metrics`: ROC/AUC, precision-recall/average precision, log loss, balanced
  accuracy, Matthews correlation, their scorers and the AUC accumulator.

### Modified Capabilities

- `scoring-contracts`: adds the probability-based scorer contract.

## Impact

- **Code:** `crates/sciencekit_metrics` gains ranking modules, scorers and an accumulator;
  `sciencekit_common` gains `SKProbabilityScorer`.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `ndarray`, `rayon`; `sciencekit_metrics` remains
  dependency-free of new external crates.
- **Downstream:** lets `SKGridSearchCV` select by AUC/log loss and lets probability
  classifiers be evaluated without manual thresholding.
- **Acceptance criteria:** small and large data, concurrency, f32/f64, companion
  `*_tests.rs` modules.
