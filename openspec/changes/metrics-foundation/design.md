# Design

## Context

See `proposal.md` — Why. The library already exposes borrowed data/target views, a scalar
bound (`SKFloat`), a central error enum, and scoring traits whose pure form takes stored
outputs and whose convenient form runs inference and delegates. Fourteen estimators
implement those traits' prerequisites, but no concrete scorer exists. `sciencekit_math`
owns the pairwise-distance kernels consumed by future neighbours, SVM and clustering code.
The specs in this change (`evaluation-metrics`, `pairwise-distances`, `error-model`,
`backend-selection`) fix the observable behaviour; this document fixes the architecture.

## Goals / Non-Goals

**Goals:**
- One crate, `sciencekit_metrics`, that owns evaluation behaviour and nothing numeric that
  already belongs to `sciencekit_math`.
- A public surface usable three ways: whole-input functions, model-bound scorer adapters,
  and mergeable accumulators for batches and parallel partitions.
- Deterministic, documented policies for label order, averaging and degenerate cases.
- O(n) single-pass evaluation with constant or class-bounded memory.

**Non-Goals:**
- The complete scikit-learn metric catalogue (ROC/AUC, calibration, multilabel, ranking);
  that stays in the later `metrics-suite` change.
- Model export/serialization; no `sciencekit_interop` crate exists yet.
- Any estimator change, Python binding, GPU backend or new external dependency.
- A sparse pairwise-distance path: NaN-aware and Chebyshev kernels are dense-only by
  contract.

## Decisions

### Decision 1 — Distances stay in `sciencekit_math`

`sk_chebyshev_distance_matrix` and `sk_nan_euclidean_distance_matrix` join the existing
`pairwise` module beside Euclidean, Manhattan and Cosine. `sciencekit_metrics` does not
duplicate or wrap them. Rationale: the `pairwise-distances` capability already owns this
behaviour, and neighbours, SVM, clustering, DBSCAN and imputation depend on `math`, not on
evaluation metrics. Putting distances in `metrics` would invert the dependency and split
one capability across two crates.

Alternative considered: mirror scikit-learn's `metrics.pairwise` namespace. Rejected —
scikit-learn's package layout is not a dependency graph; `sciencekit_math` is the numeric
foundation of every algorithm crate, and moving kernels now would break the spec boundary.

### Decision 2 — Public surface: free functions, scorer adapters, accumulators

The crate exposes three layers:

```
free functions            sk_accuracy_score, sk_confusion_matrix, sk_precision_score,
                          sk_recall_score, sk_f1_score, sk_mean_squared_error,
                          sk_root_mean_squared_error, sk_mean_absolute_error,
                          sk_r2_score
scorer adapters           SKAccuracyScorer, SKPrecisionScorer, SKRecallScorer,
                          SKF1Scorer, SKMeanSquaredErrorScorer, ... implementing
                          SKLabelScorer / SKSupervisedScorer
accumulators              SKClassificationAccumulator, SKRegressionAccumulator
```

Free functions take borrowed views (`ArrayView1`, `ArrayView2`, slices) and own nothing.
Scorer adapters hold only the metric configuration (averaging, labels, positive label) and
inherit the trait's convenient form, so no scorer re-implements inference. Accumulators
are the only stateful types.

Alternatives considered: a single `SKEvaluator` trait hierarchy (more indirection than the
contracts need) and metric objects with builders (metrics are not estimators; the builder
rule does not apply).

### Decision 3 — Scalars and return types

Supervised scorers return `F: SKFloat`, as the existing contracts require, so an f32
pipeline never detours through f64. Classification counts are `usize`; labels are `i64`.
`SKConfusionMatrix` owns a row-major `Vec<usize>` plus the resolved label list.
Standalone classification functions return `F` for symmetry with the scorer adapters.

Rationale: the contract is fixed by `scoring-contracts`; mixing `f64` returns would force
conversions at every call site and contradict the generic-float change already shipped.

### Decision 4 — Label space resolution

An explicit label list is validated (non-empty, unique, sorted internally) and defines the
class order. Without one, the metric resolves the sorted union of observed true and
predicted labels. Labels outside an explicit space produce `SKError::UnknownLabel`.
Nominal targets produce `SKError::UnsupportedRepresentation` with guidance to canonicalize
through `SKLabelTable`, matching the example scorer already in `sciencekit_common`.

Alternative considered: infer the label space from true targets only. Rejected — it would
hide a predicted class as a silent extra column and make accuracy denominators ambiguous.

### Decision 5 — Averaging and degenerate policies

`SKAveraging` has four variants: `Macro`, `Micro`, `Weighted` and
`Binary { positive_label }`. `Binary` reports one class's precision/recall/F1 and is the
explicit replacement for scikit-learn's implicit `pos_label=1`. Every zero denominator
yields zero (never NaN, never a panic). For `R²`, a zero total sum of squares yields `1.0`
when the residual is zero and `0.0` otherwise — scikit-learn's `force_finite` behaviour.

Alternatives considered: defaulting binary metrics to the positive class implicitly
(hidden state, surprising for label sets that do not contain 1) and returning NaN for
undefined precision/recall (contaminates every downstream average).

### Decision 6 — Extend the central error taxonomy, do not fork it

Three additive variants join `SKError` (already `#[non_exhaustive]`): `EmptyInput`,
`NonFiniteValue { position }` and `UnknownLabel { label }`. The metrics crate uses
`SKError` directly, so scorer adapters keep `type Error = SKError` and every algorithm can
reuse the variants. Metric-specific failures are not metric-specific in nature: an empty
observation set and a NaN value are common numeric failures.

Alternative considered: a crate-local `SKMetricsError` converting from `SKError`. Rejected
because `scoring-contracts` requires errors from the central taxonomy and the existing
example scorers already use `SKError`; a second enum would force conversions at every
boundary for no gain.

### Decision 7 — Accumulator representation and merging

Classification accumulation uses a `BTreeMap<(i64, i64), usize>` keyed by the observed
(true, predicted) pair. Observed labels need no index until finalization, `merge` is a
per-pair sum, and finalization materializes the dense `k × k` matrix in ascending label
order, optionally validating an explicit label space. Memory is `O(k²)` observed pairs and
the key order keeps the class order deterministic independent of observation order; the
per-pair map overhead is acceptable for the class cardinalities metrics target, and a dense
accumulator can replace it later if very high cardinality demands it.

Regression accumulation stores `count`, the running mean, `M2` (the Welford/Chan sum of
squared deviations from the mean) and Neumaier-compensated `Σ(y−ŷ)²` and `Σ|y−ŷ|`. The
`M2` form avoids the catastrophic cancellation of `Σy² − n·mean²` for large-magnitude `f32`
targets, and merges with the standard parallel formula
`M2 = M2a + M2b + δ²·na·nb/(na+nb)`. Every finalized metric is derived from this
constant-size state, so batches and parallel partitions produce the single-pass result
within the documented tolerance.

Alternatives considered: storing predictions (unbounded memory), a `HashMap` (nondeterministic
iteration order), and accumulating regression in `f64` (a scalar detour inconsistent with
Decision 3).

### Decision 8 — Parallelism belongs to the accumulator, not the metric function

Free functions run one sequential pass; the cost is O(n) and dominated by memory traffic.
The accumulation API is `Send + Sync`, and callers (later grid search, cross-validation,
large evaluation) partition observations, accumulate in rayon tasks and merge. This keeps
the default path allocation-free and gives concurrency without a second implementation of
every metric.

### Decision 9 — Distance algorithms and their limits

Chebyshev computes `maxᵢ |aᵢ − bᵢ|` per pair with `azip!` inside a row-parallel rayon loop;
NaN-aware Euclidean accumulates only over co-observed coordinates and applies the
`total_features / present_features` scale, returning NaN when the co-observed count is
zero. Both consume `ArrayView2` and return an owned `Array2`. Only NaN marks missingness;
other sentinels are the imputer's concern. The kernels materialize the full matrix; chunked
evaluation for memory-bounded imputation belongs to the `SKKNNImputer` change.

Alternatives considered: a generic distance trait now (premature; neighbours will justify
it) and SIMD for Chebyshev (the max reduction is not the bottleneck at this stage; the
mandatory evolution keeps the performance step for when the tree queries land).

### Decision 10 — Module layout under the 200-line rule

```
crates/sciencekit_metrics/src/
  lib.rs
  classification/   mod.rs, averaging.rs, label_space.rs, confusion.rs,
                    accuracy.rs, precision_recall_f1.rs, *_tests.rs
  regression/       mod.rs, errors.rs, r2.rs, *_tests.rs
  scorers/          mod.rs, classification.rs, regression.rs, *_tests.rs
  accumulators/     mod.rs, classification.rs, regression.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers. Every implementation file keeps its companion
`*_tests.rs` declared at the end.

## Risks / Trade-offs

- **f32 accumulation error over large inputs, including R² cancellation** → Neumaier
  compensated sums plus the Welford/Chan `M2` state in the regression accumulator; tests
  compare `f32` accuracy, MSE and R² against hand-computed `f64` references within a
  documented relative tolerance, including a large-magnitude target series.
- **O(k²) confusion memory for high-cardinality labels** → documented bound; the explicit
  label list lets callers pin the space, and the full suite can add top-k or sparse
  confusion later. Not a blocker for the current classifiers.
- **`nan_euclidean` semantics are easy to get subtly wrong** → the exact formula is in the
  spec and tests include hand-computed cases with one, several and zero co-observed
  features, plus the symmetric case.
- **Precision/recall ambiguity across multiclass conventions** → averaging is always an
  explicit parameter; no hidden default beyond `Macro`, and `Binary` requires a label.
- **New `SKError` variants** → additive and `#[non_exhaustive]`; no consumer match breaks.
- **Two crates in one change (math + metrics)** → the math additions are two kernels with
  their own tests and no API removal; review stays tractable.

## Migration Plan

1. Add `crates/sciencekit_metrics` to `[workspace.members]` and its workspace dependencies
   (`sciencekit_common`, `ndarray`, `rayon`).
2. Add the three `SKError` variants; existing code compiles unchanged because the enum is
   non-exhaustive and no existing variant changes.
3. Add the two pairwise kernels under `sciencekit_math::pairwise`, re-exported from the
   module dispatcher.
4. No data migration, feature flag, or rollback concern: the change is additive and removed
   by reverting the workspace member and the two modules.

## Open Questions

None blocking. The full metric catalogue (ROC/AUC, calibration, multilabel, ranking) and
model export remain explicit non-goals for the later `metrics-suite` and interop changes.
