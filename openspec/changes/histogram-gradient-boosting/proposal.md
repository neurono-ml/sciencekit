# Proposal

## Why

Exact CART split search sorts every feature at every node, which becomes the bottleneck on
large tabular data. The histogram variant bins each feature once, accumulates
gradient/Hessian histograms per node and grows the tree leaf-wise, the LightGBM strategy.
It keeps the boosting accuracy while making split search linear in the number of bins, and
it gives the library native categorical and missing-value handling.

## What Changes

- `SKHistGradientBoostingClassifier<F>` and `SKHistGradientBoostingRegressor<F>` in
  `sciencekit_ensemble`.
- Binned training design: at most `max_bins` quantile bins per numeric feature (default
  255), a dedicated missing-value bin, and categorical features represented by category
  bins and a `bitvec` membership mask.
- Histogram split finding: per node, accumulate gradient and Hessian sums per feature/bin,
  parallelize across features, and use histogram subtraction from the parent to avoid
  recomputation. Leaf-wise growth with `max_leaf_nodes` (default 31),
  `minimum_samples_leaf` (default 20) and `l2_regularization` (default 0.0).
- Boosting surface: `learning_rate` (default 0.1), `max_iter` (default 100), loss
  (classification `LogLoss`; regression `SquaredError`, `AbsoluteError`), `maximum_features`,
  early stopping (`validation_fraction`, `n_iter_no_change`, `tolerance`), `random_seed`,
  `backend` (default `Faer`) and `execution_mode` (default `Automatic`).
- Missing values are routed through the dedicated bin; each split learns a default
  direction for rows in that bin.
- Categorical features are declared by column index and split by partitioning categories
  into two groups using the gradient statistics and a `bitvec` mask.
- Prediction bins raw dense or sparse queries with the stored edges and traverses the
  histogram tree; no densification.
- Model accessors: binned edges, tree structures, initial prediction, learning rate,
  feature importances and the fitted iteration count.
- The in-memory binned store is the first implementation; streaming over
  `memmap2`-binned data is explicitly deferred to Wave 7.
- Validation: empty input, invalid bin/leaf/learning parameters, unknown categorical
  columns, single-class classification, non-finite values outside the missing marker and
  an invalid validation fraction.
- Explicit non-goals: `memmap2` streaming, GPU, monotonic/interaction constraints,
  quantile/poisson losses, `partial_fit` and export (Wave 7).
- Companion tests: binning edges and missing bin, hand-computed histogram gains, leaf-wise
  growth, categorical partitions, missing routing, early stopping, dense/sparse parity,
  determinism, concurrency, large data and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `histogram-gradient-boosting`: binned feature representation, histogram split finding,
  leaf-wise growth, categorical partitions, missing-value routing, early stopping and the
  classifier/regressor prediction surface.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_ensemble` gains the histogram-boosting modules; a new
  `bitvec` dependency for categorical masks and node membership.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `sciencekit_tree`, `ndarray`,
  `sprs`, `rayon`, `bitvec`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** gives the library its large-data tabular booster and the substrate for a
  future streaming/memory-mapped variant.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Streaming/memmap
  and export are deferred to Wave 7 and recorded in the PRD.
