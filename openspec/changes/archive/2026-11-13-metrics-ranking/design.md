# Design

## Context

See `proposal.md` — Why. `sciencekit_metrics` already owns the label space, confusion
matrix, precision/recall/F1 and the regression metrics. `sciencekit_common` owns the
scorer contracts. The model scalar is `F: SKFloat`; labels are canonical `i64` indexes.

## Goals / Non-Goals

**Goals:**
- Standard ranking and probability metrics with deterministic tie handling.
- Scorer adapters that fit the existing dual-form contracts.
- A mergeable AUC accumulator.

**Non-Goals:**
- Calibration curves, Brier score, ranking metrics beyond ROC/AP.
- Approximate/sketch AUC (future accelerator).
- `memmap2` and export.

## Decisions

### Decision 1 — Probability scorer contract in common

Add:

```rust
pub trait SKProbabilityScorer<F: SKFloat, M: SKClassifierPredictor<F>>
where
    Self::Error: From<M::Error>,
{
    type Error: From<SKError>;
    fn score_from_probabilities(
        &self,
        true_targets: SKTargetView<'_, F>,
        probabilities: ArrayView2<'_, F>,
    ) -> Result<F, Self::Error>;
    fn score(&self, model: &M, features: SKDataView<'_, F>, true_targets: SKTargetView<'_, F>) -> Result<F, Self::Error>;
}
```

The convenient form calls `predict_probabilities` and delegates. This keeps AUC/log loss
out of the label-only contract.

### Decision 2 — ROC/AUC

Binary AUC is computed from sorted scores with the rank-sum (Mann-Whitney) formula; ties
share the average rank. Multiclass uses one-vs-rest per class and aggregates macro, micro
or weighted. One-vs-one can be added as an explicit strategy; the first version exposes
OvR (the reference default) and documents the choice.

### Decision 3 — Precision-recall and average precision

The curve sweeps thresholds in decreasing score order, accumulating true/false positives.
Average precision is the step-wise area `Σ (R_n − R_{n−1}) P_n`. Ties use the same score
grouping as ROC.

### Decision 4 — Log loss, balanced accuracy and MCC

- Log loss clips probabilities to `[ε, 1 − ε]` (ε documented) and supports binary and
  multiclass.
- Balanced accuracy is the mean of per-class recall.
- MCC uses the standard confusion-matrix formula with a zero-denominator policy of zero.

### Decision 5 — Scorer adapters and free functions

Every metric has a free `sk_*` function and, where a model can provide the needed input, a
scorer adapter (`SKAucScorer`, `SKAveragePrecisionScorer`, `SKLogLossScorer`,
`SKBalancedAccuracyScorer`, `SKMatthewsCorrelationScorer`). The scorer tests compare the
adapter with the free function on the same data.

### Decision 6 — AUC accumulator

`SKAucAccumulator` stores `Vec<(F, i64)>`, merges by concatenation and finalizes with the
same rank-sum computation. Memory is `O(n)`; this is documented, and a future sketch-based
variant can replace it without changing the API.

### Decision 7 — Validation and concurrency

Validate lengths, finiteness, probability range and the presence of both classes for AUC.
All functions are pure and thread-safe; concurrent tests assert equality.

### Decision 8 — File layout

```
crates/sciencekit_metrics/src/ranking/
  mod.rs, roc.rs, average_precision.rs, log_loss.rs, balanced_accuracy.rs,
  matthews.rs, accumulator.rs, *_tests.rs
crates/sciencekit_common/src/scorer_traits/probability.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **AUC ties and multiclass aggregation are easy to get subtly wrong** → rank-sum with
  average ranks and reference fixtures.
- **Log loss clipping changes the value for impossible probabilities** → documented ε and a
  test.
- **AUC accumulator memory is O(n)** → documented; a sketch variant is future work.
- **One-vs-one AUC is not in the first version** → documented non-goal; OvR is the default.

## Migration Plan

Additive: new modules, scorer trait and accumulator. No existing metric changes.

## Open Questions

None blocking. One-vs-one AUC and a sketch accumulator are future changes.
