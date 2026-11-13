# Design

## Context

See `proposal.md` — Why. `sciencekit_common` provides `SKDataView`, `SKTargetView`,
`SKFeatureBlock`, `SKFloat`, `SKRandomGenerator` (from `deterministic-randomness`) and the
error taxonomy. `sciencekit_metrics` provides the scorer adapters. The crate is new.

## Goals / Non-Goals

**Goals:**
- Index-based splits that never copy the data until a fold actually needs it.
- Deterministic shuffling and stratification.
- One splitter interface usable by grid search.

**Non-Goals:**
- Time-series, repeated, leave-one-out and nested CV.
- Streaming splits and `memmap2`.
- Model fitting inside the splitters.

## Decisions

### Decision 1 — Index splits, not data splits

`SKIndexSplit { train: Vec<usize>, test: Vec<usize> }` is the unit. `sk_train_test_split`
also gathers the selected partitions into `SKFeatureBlock`/owned targets for convenience,
but the splitters themselves return indices. This keeps memory proportional to the indices
and lets grid search gather only what a fold needs.

### Decision 2 — One `SKSplitter<F>` contract with optional inputs

```rust
pub trait SKSplitter<F: SKFloat> {
    fn split(
        &self,
        features: SKDataView<'_, F>,
        targets: Option<SKTargetView<'_, F>>,
        groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError>;
}
```

Plain K-fold ignores targets/groups; stratified uses targets; group uses groups. Grid search
passes whichever its estimator family needs. This avoids three parallel splitter traits and
still documents the required input per implementation.

### Decision 3 — Seeded Fisher-Yates

All shuffles use `SKRandomGenerator` from `deterministic-randomness`; the generator is
seeded by `random_seed` (default 0). The same seed and row count produce the same order
across dense and sparse inputs.

### Decision 4 — Stratification

For classification targets, canonicalize labels (integer or nominal), group row indices by
class, shuffle each class with the seeded generator, then deal the class members
round-robin into the folds. This preserves proportions and is deterministic. A class with
fewer members than the fold count is rejected rather than silently dropped.

### Decision 5 — Group K-fold

Distinct groups are shuffled with the seed and dealt round-robin into the folds; every row
of a group goes to the same fold. The split count defaults to the number of distinct
groups. A group count smaller than an explicit split count is rejected.

### Decision 6 — Row gathering

`sk_gather_rows` selects rows from a dense `Array2` or CSR `CsMat` into an owned
`SKFeatureBlock`; the sparse path copies only stored entries and keeps CSR. Target
gathering produces owned continuous/integer/nominal arrays. The helper is shared by
`sk_train_test_split`, `sk_cross_val_score` and grid search.

### Decision 7 — Cross-validation scoring

`sk_cross_val_score` is generic over the estimator family and the scorer policy. It fits on
the train indices, scores on the test indices and returns an `Array1<F>` of scores in fold
order. Folds are independent and run in parallel; results are collected by fold index. The
classifier/regressor variants use the matching scorer enum from grid search, or a closure
for a custom metric.

### Decision 8 — Validation and observability

Splitters validate fractions, counts, class/group cardinality, lengths and finiteness
before producing indices. `sk_train_test_split` and `sk_cross_val_score` record spans with
the backend of the fitted estimators; splitters themselves have no backend.

### Decision 9 — File layout

```
crates/sciencekit_model_selection/src/
  lib.rs
  index_split.rs
  gather.rs
  split/{mod.rs, kfold.rs, stratified.rs, group.rs, builder.rs, *_tests.rs}
  train_test_split/{mod.rs, split.rs, stratified.rs, *_tests.rs}
  cross_validation/{mod.rs, classifier.rs, regressor.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Index vectors cost O(n) per fold** → inherent to cross-validation; training/test sets
  are disjoint and only one fold is materialized at a time.
- **Stratification with rare classes** → rejected when the class is smaller than the fold
  count; documented.
- **Gathering copies rows** → unavoidable for shuffled splits; sparse gathering never
  densifies.
- **One splitter trait with optional inputs** → slightly less type-safe than three traits;
  validation catches missing targets/groups.
- **Parallel folds multiply memory** → the execution mode bounds the grain; document the
  peak.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking. Time-series and repeated CV are future changes.
