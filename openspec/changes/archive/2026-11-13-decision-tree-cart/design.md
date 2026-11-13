# Design

## Context

See `proposal.md` — Why. There is no tree crate yet. `sciencekit_common` provides the
builder state, label table, data views, errors and observability; `sciencekit_math`
provides the reductions and sparse views; `sciencekit_metrics` provides the acceptance
metrics. `sprs` already exposes CSR and CSC views, so sparse split search does not require
a new dependency.

## Goals / Non-Goals

**Goals:**
- A correct, deterministic CART implementation for classification and regression.
- Dense and sparse fit/predict without densification.
- A tree structure reusable by random forests, bagging, boosting and isolation forests.
- Feature importances and probability estimates.

**Non-Goals:**
- Missing-value routing (rejected in this change).
- Pruning, incremental/out-of-core fitting, `memmap2` backing and persistence/export.
- Histogram/binned splits (W4.5), ensembles (W4.2–W4.4).

## Decisions

### Decision 1 — New `sciencekit_tree` crate with a flat node array

The tree is a `Vec<SKTreeNode<F>>` with a root index. Each node stores the split feature
(or a leaf sentinel), the threshold, the left/right child indices, the sample count, the
impurity and the leaf value (class counts or regression value). A flat array avoids
pointer chasing, makes prediction cache-friendly, is trivially `Send + Sync` and is the
structure the ensembles will copy.

Alternative considered: `Box`-linked nodes (allocation-heavy, harder to snapshot) and an
adjacency-list graph (unnecessary for a binary tree).

### Decision 2 — Split search by sorted feature scans

At each node the splitter evaluates candidate features in parallel with rayon. For a
dense design it sorts the node's sample indices by the feature value, then scans the
sorted order accumulating the criterion statistics and evaluates every valid threshold.
For a sparse design it uses a CSC view of the training design (built once at fit, never a
dense copy) and merges the stored entries with the implicit zeros, so the scan sees the
true value of every node sample. The best split is the one with the largest impurity
decrease; ties break by smaller feature index, then smaller threshold.

Alternative considered: per-node dense submatrices (would densify sparse data and blow up
memory) and histogram splits (that is W4.5; CART keeps exact splits).

### Decision 3 — Criteria

- Classification: Gini impurity `1 - Σ p²` and entropy `-Σ p log p`, over the node class
  counts. The leaf predicts the majority class and stores the normalized counts for
  probabilities.
- Regression: squared error around the mean, with the leaf predicting the mean; absolute
  error around the median, with the leaf predicting the median. The absolute-error split
  search sorts the node responses and uses prefix sums to evaluate thresholds.

Alternatives considered: Poisson/quantile criteria (out of scope) and multi-output trees
(out of scope).

### Decision 4 — Constraints and feature subsampling

`maximum_depth`, `minimum_samples_split`, `minimum_samples_leaf` and
`maximum_features` (`All`, `Count`, `Fraction`, `Sqrt`, `Log2`) bound the growth.
`splitter = Best` evaluates all selected features; `splitter = Random` draws a random
threshold per selected feature. The seeded SplitMix64 generator (the pattern already used
by SGD) makes `Random` and feature subsampling reproducible.

### Decision 5 — Representation parity

Fit accepts dense or borrowed CSR; a CSC copy of a sparse design is built once for
column-wise split search, which is a re-indexing, not a densification. Prediction accepts
dense or sparse views and dispatches once; a sparse row is walked through the tree by
reading only its stored entries (an absent entry is the observed zero). No path creates a
dense copy of a sparse design.

### Decision 6 — Determinism and concurrency

Feature order, threshold order and tie-breaking are deterministic; the seeded generator
covers the randomized splitter and feature subsampling. Prediction rows are independent
and run through rayon, collecting in order. The fitted model is immutable and
`Send + Sync`.

### Decision 7 — Feature importances

Each split contributes `n_samples · impurity_decrease` to its feature; the vector is
normalized to sum to one when the tree splits at least once, and is zero otherwise. This
is the standard impurity-based importance and matches scikit-learn's normalization.

### Decision 8 — Validation and observability

Builders expose `backend` (default `Faer`) and `execution_mode` (default `Automatic`);
fit transfers both into the model and every span records the backend. Errors:
`EmptyInput`, `InvalidHyperparameter` (depth/split/leaf/feature parameters),
`ShapeMismatch`, `NonFiniteValue`, `UnsupportedRepresentation` if a representation cannot
be handled, and `UnsupportedOperation` for a single-class classification target if the
classifier cannot form a split.

### Decision 9 — File layout

```
crates/sciencekit_tree/src/
  lib.rs
  node.rs
  impurity.rs
  splitter/{mod.rs, best.rs, random.rs, candidate.rs, *_tests.rs}
  design/{mod.rs, dense.rs, sparse.rs, *_tests.rs}
  classifier/{mod.rs, builder.rs, core_implementation.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  regressor/{mod.rs, builder.rs, core_implementation.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **`O(d · n log n)` split search per node** → parallel feature scans and exact CART; the
  histogram variant arrives in W4.5 for large data.
- **Sparse split search is more expensive than dense** → CSC indexing once at fit and
  stored-entry scans; no densification.
- **Absolute-error split search is slower** → documented; squared error is the default.
- **Deep trees overfit** → depth/leaf constraints and the ensemble wave.
- **No missing-value routing** → NaN is rejected; a later change adds routing.
- **Feature importances are impurity-based and biased toward high-cardinality features**
  → documented, matching the reference.

## Migration Plan

Additive: a new crate and workspace membership. No existing API changes.

## Open Questions

None blocking. Missing-value routing, pruning and the histogram splitter are later
changes; persistence/export and memory-mapped backing are Wave 7.
