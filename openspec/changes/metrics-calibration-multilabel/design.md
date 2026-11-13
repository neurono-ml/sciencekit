# Design

## Context

See `proposal.md` — Why. `sciencekit_metrics` already owns the label space, the confusion
matrix, precision/recall/F1, the regression errors and (through `metrics-ranking`) the
score-based metrics. Multi-label targets reuse the same canonical `i64` label space, now
interpreted as a set per row. Probabilities arrive as a two-column binary matrix or as a
`(rows, classes)` multiclass matrix. Scalar results are `F: SKFloat`.

The reference behaviour for every metric is scikit-learn: `calibration_curve`,
`brier_score_loss`, `hamming_loss`, `jaccard_score`, `accuracy_score`
(subset accuracy), `precision_recall_fscore_support` with `average` and
`label_ranking_average_precision_score`. Expected calibration error is not part of
scikit-learn; the reference is the standard Guo et al. (2017) definition, documented here.

## Goals / Non-Goals

**Goals:**
- Calibration curves with uniform and quantile binning and typed bin data.
- Top-label reliability data for multiclass probability matrices.
- Brier score and expected calibration error with a configurable bin count.
- Multi-label losses and scores over dense and CSR inputs, with the four averaging modes.
- Exact scikit-learn parity on the documented semantics and zero-division policy.

**Non-Goals:**
- Calibration maps (isotonic, Platt/sigmoid, temperature scaling).
- Brier score decomposition (reliability, resolution, uncertainty).
- NDCG, coverage error, ranking loss and multilabel confusion matrices.
- Streaming accumulators for the new metrics.
- Multiclass Brier score and multiclass ECE variants beyond top-label.

## Decisions

### Decision 1 — Free functions and typed structs, no builders

Metrics are not estimators: they have no fit state, no backend selection and no
`execution_mode`. Following `metrics-foundation` and `metrics-ranking`, every metric is a
free `sk_*` function returning `Result<_, SKError>` and validating its own inputs. The
calibration results are typed public structs so callers never index parallel vectors:

```rust
pub struct SKCalibrationBin<F> {
    pub lower_bound: F,
    pub upper_bound: F,
    pub number_of_samples: usize,
    pub fraction_of_positives: F,
    pub mean_predicted_probability: F,
}

pub struct SKCalibrationCurve<F> {
    pub bins: Vec<SKCalibrationBin<F>>,
    pub strategy: SKCalibrationStrategy,
    pub number_of_samples: usize,
}
```

`SKCalibrationStrategy` is `Uniform { number_of_bins }` or `Quantile { number_of_bins }`.
A builder would add nothing here and would contradict the crate's existing style.

### Decision 2 — Bin construction and edge inclusion

- **Uniform**: `number_of_bins` equal-width bins over `[0, 1]`. The lower edge of a bin is
  exclusive and the upper edge inclusive, except the first bin, which includes `0`. Values
  outside `[0, 1]` are rejected as invalid probabilities. Empty bins are retained with
  `number_of_samples = 0` and `fraction_of_positives`/`mean_predicted_probability` set to
  zero, so curve length always equals `number_of_bins`; this is a documented divergence
  from scikit-learn, which drops empty bins.
- **Quantile**: edges are the empirical quantiles of the predicted probabilities at
  `i / number_of_bins`. Duplicate edges (many identical scores) are collapsed, so fewer
  than `number_of_bins` bins may be returned; a curve with a single distinct probability
  yields one bin. This matches scikit-learn's duplicate-edge behaviour.

Per-bin summaries are computed with `azip!()`/iterators and a parallel row reduction;
bins are then finalized in index order so the result is deterministic.

### Decision 3 — Top-label reliability data

`sk_top_label_calibration_curve(probabilities, true_targets, strategy)` assigns each row
its argmax class and takes `max(probability)` as the confidence. The label is correct when
the argmax equals the true class. The resulting correctness indicators and confidences feed
the same curve builder as the binary case, so a reliability diagram for a multiclass model
comes from one documented code path. Ties in the argmax resolve to the smallest class
index, matching the label canonicalization order.

### Decision 4 — Brier score and expected calibration error

- `sk_brier_score(true_targets, probabilities)` is the mean of `(p − y)^2` over binary
  rows, with `y ∈ {0, 1}` and `p` the positive-class probability. It returns
  `ShapeMismatch` for a probability matrix that is not `(n, 2)` and `InvalidTarget` for
  labels outside the two classes.
- `sk_expected_calibration_error(true_targets, probabilities, strategy)` is the top-label
  ECE: `Σ_b (|B_b| / n) · |accuracy(B_b) − confidence(B_b)|`, with the same bins as the
  curve. The binning strategy is the same enum, so ECE and the reliability diagram always
  agree when configured identically.
- Both reject non-finite probabilities and probabilities outside `[0, 1]`.

### Decision 5 — Multi-label representation dispatch

Public functions accept `SKDataView`-style dense/sparse inputs for the indicator matrix and
a dense target matrix (or the canonical target view) for the truth. Dispatch happens once
per call:

- dense indicator `ArrayView2<F>` → row slices via `axis_iter`;
- CSR `CsrView<F>` (or `CsrMatrix`) → `outer_iterator()` rows, reading only stored entries.

No branch materializes a dense copy of a sparse matrix. An indicator matrix must contain
only `0` and `1` (a value close to one within the documented tolerance is accepted and
treated as set); any other value returns `InvalidTarget` naming the row and column. The
same rule applies to dense and CSR so the representations cannot disagree on validity.

Set operations use per-row true-positive, false-positive and false-negative counts
accumulated as `usize`. Dense and sparse paths are separate loops over the same counters and
are asserted equal in tests within `1e-12` for `f64` and `1e-5` for `f32`.

### Decision 6 — Averaging semantics

`SKMultilabelAverage` is `Binary`, `Micro`, `Macro`, `Weighted` or `Samples`, mirroring
`sklearn.metrics`:

- **Binary**: per-label score for label 1 (the first label column); the returned per-label
  vector still contains every label.
- **Micro**: pool true positives, false positives and false negatives across all labels,
  then compute one precision/recall/F1.
- **Macro**: unweighted mean of the per-label scores.
- **Weighted**: support-weighted mean, where support is the number of true positives plus
  false negatives for that label.
- **Samples**: compute the metric per row and average over rows.

A label whose denominator is zero contributes `0` to the average, matching scikit-learn's
default `zero_division=0`; the policy is documented on the function and asserted by a test.
`sk_jaccard_score` uses the same enum and the same zero-division policy. Hamming loss is
`(FP + FN) / (rows · labels)` and subset accuracy is the fraction of rows whose predicted
set equals the true set; neither takes an averaging mode.

`sk_label_ranking_average_precision(predicted_scores, true_indicator)` averages, per row,
the precision at each rank occupied by a true label divided by the number of true labels;
a row with no true label is rejected as `InvalidTarget`.

### Decision 7 — Result types for the multilabel block

`sk_multilabel_precision_recall_f1` returns `SKMultilabelReport<F>` with per-label
`SKLabelScore<F> { label, precision, recall, f1, support }` and the aggregate values for the
requested averaging mode, so a caller can print a full report without repeating the
computation. The free functions for Hamming, Jaccard, subset accuracy and LRAP return `F`
directly because their result is scalar.

### Decision 8 — Validation, concurrency and layout

All functions are pure and thread-safe; a concurrent test asserts equality with the
sequential result. Validation errors are the existing typed `SKError` variants:
`EmptyInput`, `ShapeMismatch`, `InvalidTarget`, `InvalidHyperparameter` and
`NonFiniteValue` where applicable.

```
crates/sciencekit_metrics/src/calibration/
  mod.rs, curve.rs, binning.rs, top_label.rs, brier.rs, expected_calibration_error.rs,
  validation.rs, *_tests.rs
crates/sciencekit_metrics/src/multilabel/
  mod.rs, representation.rs, loss.rs, jaccard.rs, subset_accuracy.rs,
  precision_recall_f1.rs, label_ranking.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines and tests
live in companion `*_tests.rs` modules declared at the end.

## Risks / Trade-offs

- **Empty-bin retention diverges from scikit-learn** → deliberate, documented and tested;
  it keeps the curve indexable by bin number and simplifies plotting.
- **Quantile duplicate edges change the bin count** → matches scikit-learn; documented and
  tested with tied scores.
- **Averaging modes are subtle at the edges (absent labels, zero denominators)** →
  hand-computed parity fixtures for each mode plus a zero-division test.
- **Sparse indicator validation could scan every stored value** → O(nnz), same order as the
  metric itself; no densification.
- **ECE definition varies across the literature** → top-label ECE with the documented
  `Σ (|B|/n)|acc − conf|` formula and the same bins as the curve.
- **Indicator tolerance can hide a bad encoding** → the tolerance is documented and any
  value outside it is rejected with the offending row and column.

## Migration Plan

Additive: new modules and free functions. No existing metric changes and no breaking
changes. The catalog row moves from "Planned" to an implemented entry only when this change
lands.

## Open Questions

None blocking. Calibration maps (isotonic/sigmoid/temperature scaling), multiclass Brier
decomposition and streaming accumulators are future changes.
