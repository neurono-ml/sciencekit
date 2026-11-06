# Design

## Context

See `proposal.md` — Why. `sciencekit_tree` (W4.1) supplies the flat node structure and
dense/sparse traversal. `sciencekit_common` gains `SKOutlierPredictor<F>`. The detector
uses random splits rather than impurity-driven CART splits, so the tree crate's split
search is not reused; only the node representation/traversal is shared.

## Goals / Non-Goals

**Goals:**
- Exact isolation-forest scoring with a deterministic seed.
- Dense and sparse parity without densification.
- A reusable outlier contract and contamination offset.

**Non-Goals:**
- LOF, One-Class SVM, missing values, streaming, GPU, `memmap2`, export.
- Missing-value routing in isolation trees.

## Decisions

### Decision 1 — New `sciencekit_outlier` crate

The crate owns the detector and its isolation-tree builder. `SKOutlierPredictor<F>` and
the one-byte `SKOutlierLabel { Inlier, Outlier }` are added to `sciencekit_common` and
re-exported, mirroring the other model-side predictors. Integer `-1`/`1` exists only at
the interop boundary.

### Decision 2 — Random isolation trees on the shared node array

Each tree is built on a seeded subsample of at most `maximum_samples` rows. A node with
more than one sample picks a random feature (from `maximum_features`) and a random
threshold uniformly between the feature's observed min and max, then partitions the rows.
Growth stops at one sample or the depth limit `ceil(log2(max_samples))`. Nodes use the flat
`SKTreeNode` array (feature/threshold/left/right/size); leaves store their sample count.

Alternative considered: reusing the CART splitter (wrong objective: isolation needs random
splits, not impurity decrease).

### Decision 3 — Path length and anomaly score

For a row, the path length in a tree is its depth plus `c(leaf_size)`, where
`c(n) = 2H(n−1) − 2(n−1)/n` and `H` is the harmonic number. The forest score is
`2^(−E[h] / c(max_samples))`. Rows isolated at shallower depths score higher. The model
exposes `score_samples` as the negation of the anomaly score minus the offset convention
documented in the spec, and `decision_function` as `score_samples − offset`.

### Decision 4 — Contamination

`contamination` is either `Auto` (a fixed documented offset) or a fraction in `(0, 0.5]`.
An explicit fraction sets the offset to the corresponding quantile of the training
`score_samples`; `Auto` uses the fixed offset. The offset is stored on the model.

### Decision 5 — Representation parity

Dense input uses the dense tree builder; sparse input uses the CSC view for feature
min/max and split scans, without densification. Prediction dispatches once and walks the
tree through the shared node traversal.

### Decision 6 — Validation and observability

Builder validation: trees > 0, sample size in `1..=n`, contamination in range, features in
`(0, 1]`. Fit validates non-empty input and finiteness; prediction validates the feature
count. Spans record the backend; fit transfers it to the model.

### Decision 7 — File layout

```
crates/sciencekit_outlier/src/
  lib.rs
  isolation_forest/{mod.rs, builder.rs, config.rs, tree.rs, fit.rs, score.rs, model.rs, predict.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Random splits make the score stochastic** → per-tree seeds derived from `random_seed`; a
  fixed seed gives identical results.
- **`maximum_samples` caps tree size** → standard isolation-forest behavior; documented.
- **Contamination is a user prior** → exposed offset and documented `Auto` behavior.
- **Sparse split search scans stored entries** → no densification; performance documented.
- **No missing-value routing** → NaN is rejected; a later change can add it.

## Migration Plan

Additive: new crate, new common trait and workspace member. No existing API changes.

## Open Questions

None blocking. Missing-value routing and streaming are future changes.
