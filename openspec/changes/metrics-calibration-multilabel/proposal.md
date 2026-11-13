# Proposal

## Why

`metrics-foundation` shipped the label-space metrics and `metrics-ranking` shipped the
score-based ones, but the crate still cannot answer two questions every probability model
and every multi-label workflow asks:

1. **Are the predicted probabilities trustworthy?** There is no calibration curve, no
   Brier score and no expected calibration error, so a well-ranked but badly calibrated
   classifier cannot be diagnosed. scikit-learn's `calibration_curve` and
   `brier_score_loss` are the reference behaviour.
2. **How good is a multi-label prediction?** Hamming loss, Jaccard score, subset accuracy,
   the four averaging modes of precision/recall/F1 and label-ranking average precision are
   the standard scikit-learn multi-label surface, and none of them exists yet.

Both gaps are visible in the public catalog (`docs/src/algorithms.md`, "Calibration and
multilabel metrics") and were declared non-goals of `metrics-ranking`; this change turns
that row into a concrete, testable plan.

## What Changes

- Add `sk_calibration_curve` with two binning strategies, uniform and quantile, returning
  the typed `SKCalibrationCurve` composed of `SKCalibrationBin` values (lower and upper
  bound, sample count, fraction of positives and mean predicted probability).
- Add top-label reliability data: a function that reduces a multiclass probability matrix
  to (confidence, correctness) pairs and returns the same `SKCalibrationCurve`, so
  reliability diagrams are drawn from typed data instead of ad-hoc tuples.
- Add `sk_brier_score` for binary targets and probabilities, plus `sk_expected_calibration_error`
  with a configurable number of bins and strategy.
- Add multilabel losses and scores over dense indicator matrices and CSR matrices, never
  densifying sparse input:
  - `sk_hamming_loss`;
  - `sk_jaccard_score` with `binary`, `micro`, `macro`, `weighted` and `samples` averaging;
  - `sk_subset_accuracy`;
  - `sk_multilabel_precision_recall_f1` returning a typed per-label and aggregate result for
    `micro`, `macro`, `weighted` and `samples` averaging;
  - `sk_label_ranking_average_precision`.
- Document, for every metric, its scikit-learn counterpart, its averaging semantics, its
  zero-division policy and its dense/sparse equivalence tolerance.
- Validation: empty input, shape or length mismatch, non-indicator values in a multilabel
  matrix, empty label set, a single class for calibration, invalid bin count, invalid
  strategy and non-finite probabilities.
- Explicit non-goals: multiclass and multilabel calibration maps (isotonic, sigmoid,
  temperature scaling), Brier decomposition, NDCG/coverage/ranking loss, multilabel
  stratification and streaming accumulators.
- Companion tests: hand-computed calibration fixtures, empty and duplicate bins,
  dense/sparse equivalence, f32/f64 precision, large data and concurrency.

No existing metric behaviour changes and no breaking changes.

## Capabilities

### New Capabilities

- `metrics-calibration-multilabel`: calibration curves and bins, top-label reliability,
  Brier score, expected calibration error and the multilabel losses/scores with their
  averaging semantics and dense/sparse parity.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_metrics` gains `calibration/` and `multilabel/` modules with
  typed result structs; `sciencekit_common` is unchanged.
- **APIs:** additive free functions and result types; no builder is needed because metrics
  are not estimators.
- **Dependencies:** `ndarray`, `sprs`, `rayon`, `sciencekit_common`; no new external crate.
- **Downstream:** calibration diagnostics become available to every probability classifier
  and to the documentation examples; the multi-label block is no longer missing from the
  scikit-learn taxonomy.
- **Acceptance criteria (PRD §8.7):** small and large data, concurrency, f32/f64, CSR
  without densification and companion `*_tests.rs` modules.
