# Design

## Context

See `proposal.md` — Why. The boosting substrate and loss contract arrive with
`gradient-boosting` (W4.4). `sciencekit_tree` supplies the flat node structure. `bitvec`
is a new small dependency used for categorical membership and node masks. `memmap2` is
deferred to Wave 7, so the binned store is in memory in this change.

## Goals / Non-Goals

**Goals:**
- Split search linear in the number of bins, not in the number of samples.
- Native missing-value routing and categorical partitions.
- A binned store that a future memory-mapped/streaming variant can replace.
- Reuse the gradient/Hessian and Newton-leaf contract from W4.4.

**Non-Goals:**
- `memmap2` streaming, GPU, monotonic/interaction constraints.
- Quantile/Poisson losses, `partial_fit`.
- Persistence/export.

## Decisions

### Decision 1 — Column-major binned store

Fit builds a `SKBinnedDataset`:

```
n_features
max_bins
edges: Vec<Vec<F>>                // quantile edges per numeric feature
bins: Vec<u8>                     // column-major: feature block of n_samples
missing_bin: u8                   // dedicated bin, e.g. max_bins
categorical: BitVec               // feature is categorical
categories: Vec<Vec<i64>>         // category code -> bin per categorical feature
```

Column-major storage makes histogram accumulation over a node's sample indices
cache-friendly and parallel across features. Prediction bins a raw row on the fly; it
does not need a stored binned matrix for the query. A sparse query's implicit zero is
binned as the observed zero value; missing values only exist in dense input and use the
dedicated bin.

Alternative considered: row-major storage (better for row-wise prediction but strided for
feature-parallel histograms) and storing both layouts (2× memory for no gain).

### Decision 2 — Quantile binning

For each numeric feature, sort the observed values (subsampled deterministically when the
column is very large), choose at most `max_bins - 1` interior quantile edges, and assign
values by binary search. The dedicated missing bin is reserved; categorical features map
their observed category codes to bins and keep an explicit unknown-category bin.

### Decision 3 — Histogram split finding

For a node, every feature accumulates `(G_bin, H_bin, count_bin)` over its sample indices.
Features are processed in parallel with rayon. Numeric splits scan the bins left to right,
maintaining prefix `G`/`H`, and evaluate the gain:

```
gain = G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)
```

subject to `minimum_samples_leaf` on both sides. The missing bin is evaluated as part of
the scan and its best direction is stored as `default_left`. Categorical features sort
their category bins by `G/H`, scan prefix partitions and store the left partition as a
`BitVec`.

### Decision 4 — Leaf-wise growth with histogram subtraction

A priority queue orders splittable leaves by gain. The highest-gain leaf is split until
`max_leaf_nodes` or no valid split remains. The parent histogram is kept during the split;
the smaller child is accumulated directly and the larger child is obtained by subtracting
the smaller from the parent. This keeps histogram work proportional to the smaller child.

The tree uses the flat `SKTreeNode` array from `sciencekit_tree`; the histogram booster
does not reuse the CART splitter, only the node representation and traversal.

### Decision 5 — Missing-value routing

Rows in the missing bin are included in the scan. If the best split puts the missing bin
left, `default_left = true`; if right, false. A node with no missing rows has no learnable
direction: both candidates tie and the scan keeps the first one it evaluates, which is the
right direction (`default_left = false`), so prediction stays well-defined and deterministic.

### Decision 6 — Categorical partitions

A categorical feature's bins are the observed categories plus an unknown bin. The split
sorts categories by their gradient-to-Hessian ratio and scans prefix partitions; the best
partition is stored as a `BitVec` with one bit per category bin. Prediction maps the raw
category to its bin and follows the bit; an unknown category uses the documented unknown
direction.

### Decision 7 — Boosting reuse

The stage loop, loss contract, initial prediction, learning rate, early stopping and
Newton leaf values are the W4.4 substrate. This change only replaces the split finder and
the tree growth policy; it does not duplicate the loss code.

### Decision 8 — Prediction

Prediction bins each raw row with the stored edges (binary search) or category map, then
walks the histogram tree using the numeric threshold or categorical `BitVec`. Sparse query
rows are read through their stored entries; an absent entry is the observed zero. The
result is added to the initial prediction and passed through the loss link.

### Decision 9 — Validation and observability

Builder defaults and validation follow the proposal. `fit` transfers backend/execution
mode to the model and every span records the backend. Errors: `EmptyInput`,
`InvalidHyperparameter`, `ShapeMismatch`, `NonFiniteValue` (outside the missing marker),
`UnsupportedRepresentation` (a categorical index outside range) and the central tree
errors.

### Decision 10 — File layout

```
crates/sciencekit_ensemble/src/histogram_boosting/
  mod.rs
  binning/{mod.rs, edges.rs, dataset.rs, categorical.rs, *_tests.rs}
  histogram/{mod.rs, accumulate.rs, subtract.rs, gain.rs, *_tests.rs}
  growth/{mod.rs, leaf_wise.rs, node.rs, *_tests.rs}
  classifier.rs, regressor.rs, builder.rs, model.rs, fit.rs, predict.rs, *_tests.rs
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Binning loses precision** → documented; `max_bins` controls the trade-off and exact
  CART remains available in `sciencekit_tree`.
- **Column-major store doubles memory versus row-major for some access patterns** → chosen
  for feature-parallel histograms; the store is `u8` per cell, so it is compact.
- **Categorical split ordering can overfit** → `minimum_samples_leaf` and the
  `l2_regularization` term apply to every split; a future `min_data_in_leaf` per category
  can be added.
- **Histogram subtraction adds bookkeeping** → tests compare it with direct accumulation.
- **Missing-value routing is dense-only** → CSR has no missing marker; documented.
- **In-memory binned store** → the future memmap/streaming variant replaces the store
  behind the same interface.

## Migration Plan

Additive: new estimators in `sciencekit_ensemble` plus a `bitvec` dependency. No existing
API changes.

## Open Questions

None blocking. Monotonic constraints, interaction constraints and the memory-mapped
streaming store are future changes.
