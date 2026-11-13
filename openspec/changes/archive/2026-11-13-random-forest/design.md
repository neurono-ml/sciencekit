# Design

## Context

See `proposal.md` — Why. `sciencekit_tree` (W4.1) supplies `SKDecisionTreeClassifier`,
`SKDecisionTreeRegressor`, the flat node structure, feature importances and dense/sparse
split search. `sciencekit_ensemble` is a new crate for the wave's aggregators. The seeded
generator pattern already exists in `sciencekit_linear_model` (SplitMix64) and will be
shared by the ensemble crate.

## Goals / Non-Goals

**Goals:**
- A forest whose result is independent of thread count and repeated fits.
- Correct OOB estimation without a second full prediction pass.
- Reuse the tree crate rather than duplicating CART.

**Non-Goals:**
- Extremely randomized trees, missing-value routing, streaming/`partial_fit`.
- `memmap2` backing, persistence and export (Wave 7).
- A generic bagging mechanism (that is W4.3).

## Decisions

### Decision 1 — Forest lives in `sciencekit_ensemble`, trees stay in `sciencekit_tree`

`SKRandomForestClassifier`/`Regressor` own a `Vec<SKDecisionTreeModel<F>>` and delegate
all split logic to the tree crate. The forest adds bootstrap sampling, parallel fitting,
aggregation, OOB and forest-level validation. Alternative: subclassing the tree (no
inheritance in Rust, and the tree model is already the right unit).

### Decision 2 — Deterministic per-tree bootstrap

A tree `i` receives `seed_i = splitmix64(global_seed ^ i)`. The sampler draws `n` row
indices with replacement (or all rows when `bootstrap = false`). The OOB mask is the
complement of the drawn indices. Because the seed depends only on the global seed and the
index, fitting with rayon in any order produces the same forest.

For sparse input the sampled rows are materialized into a new CSR (a row copy, never a
densification); dense input materializes a new `Array2`. This keeps the tree builder's
contiguous-view contract.

### Decision 3 — Feature subsampling is delegated to the tree splitter

`maximum_features` (default `Sqrt`) is passed to every tree builder together with the
tree seed. The tree selects features per node, so the forest does not duplicate the
selection logic. `All`, `Count`, `Fraction`, `Sqrt` and `Log2` are inherited from the
tree.

### Decision 4 — Parallel lock-free fitting

Trees are independent: `(0..number_of_trees).into_par_iter().map(fit_tree).collect()`
each writes its own slot. No mutex, no shared accumulator, no reduction order dependency.
The method respects the configured `SKExecutionMode`; under `Automatic` the tree count and
sample size determine the grain.

### Decision 5 — Aggregation

The classifier stores the canonical class table once (from the full targets) and averages
per-tree probabilities, predicting the largest average with the smallest-index tie-break.
The regressor averages responses. Both are `O(n_classes)` per row and parallelize over
rows.

### Decision 6 — OOB estimation

Each tree records its OOB row indices. After fitting, the model walks the OOB rows and
averages predictions only from the trees that did not see the row. Classification uses
accuracy; regression uses R², computed through `sciencekit_metrics`. Rows with no OOB tree
are excluded and the model exposes the count of scored rows; a forest with no scored rows
reports `UnsupportedOperation` rather than a fabricated score. `oob_score` without
bootstrap is rejected at build time.

### Decision 7 — Feature importances

The forest importance is the arithmetic mean of the per-tree normalized importances,
re-normalized to sum to one when at least one tree splits. The tree crate already computes
the per-tree vector.

### Decision 8 — Representation parity

The forest never densifies: dense input goes to dense trees; sparse input goes to sparse
trees after row sampling. Prediction dispatches once and uses the tree model's
dense/sparse traversal. Mixed prediction is not applicable because a tree model has one
training representation, but a sparse model can predict on dense and vice versa through
the tree contract.

### Decision 9 — Validation and observability

Builder defaults and validation are the tree defaults plus `number_of_trees`,
`bootstrap`, `maximum_samples`, `oob_score` and the seed. `fit`/`predict` record the
backend; `fit` transfers backend/execution mode into the model. Errors:
`EmptyInput`, `InvalidHyperparameter`, `ShapeMismatch`, `NonFiniteValue`, `NotConverged`
(not used by the forest) and the tree's errors converted into the central taxonomy.

### Decision 10 — File layout

```
crates/sciencekit_ensemble/src/
  lib.rs
  bootstrap/{mod.rs, sampler.rs, oob.rs, *_tests.rs}
  random_forest/{mod.rs, classifier.rs, regressor.rs, builder.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Bootstrap materializes a copy per tree** → unavoidable for contiguous tree builders;
  memory is one tree's sample at a time under the parallel map.
- **Parallel fit memory grows with tree count** → the execution mode bounds the grain;
  document the peak.
- **OOB coverage can be incomplete for small `n`** → the model reports scored rows and
  errors when none exist.
- **Forest importance inherits the tree bias toward high-cardinality features** →
  documented.
- **`bootstrap = false` disables OOB** → rejected at build, not silently ignored.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking. Extremely randomized trees and missing-value routing are future changes.
