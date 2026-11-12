# Proposal

## Why

Wave 4 starts with the decision tree because every ensemble in the wave is a tree
aggregator. The library currently has no recursive partitioning estimator, no impurity
driven split search and no tree-shaped model to reuse. CART is the shared foundation for
`SKRandomForest`, `SKBagging`, `SKGradientBoosting` and `SKHistGradientBoosting`, and it
also unlocks `SKIsolationForest` in Wave 5.

## What Changes

- New sub-crate `sciencekit_tree` with `SKDecisionTreeClassifier<F>` and
  `SKDecisionTreeRegressor<F>`, each with a distinct fitted model.
- Builder-only surface: `criterion` (classification `Gini` default, `Entropy`; regression
  `SquaredError` default, `AbsoluteError`), `maximum_depth` (default unlimited),
  `minimum_samples_split` (default 2), `minimum_samples_leaf` (default 1),
  `maximum_features` (`All` default, `Count`, `Fraction`, `Sqrt`, `Log2`),
  `splitter` (`Best` default, `Random`), `random_seed` (default 0),
  `class_weight` (`None` default, `Balanced`, explicit), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`).
- CART fitting: binary recursive partitioning, impurity decrease maximization, dense and
  sparse designs, array-based tree storage with contiguous node and child arrays.
- Dense and sparse representation parity: fit accepts dense or borrowed CSR; sparse split
  search scans stored entries column-wise without densifying. Prediction accepts dense and
  sparse views through the same boundary.
- Classification: class-probability estimates from the leaf class distribution,
  `predict_labels` through the canonical class table, deterministic class ordering and
  tie-breaking.
- Regression: leaf mean (squared error) or leaf median (absolute error) prediction.
- Model accessors: node count, depth, feature importances and the fitted class table.
- Validation: empty input, invalid depth/split/leaf parameters, feature-count mismatch,
  non-finite values (missing values are rejected in this change and handled by a later
  missing-value change), single-class classification and unsupported criterion.
- Explicit non-goals: missing-value routing, `partial_fit`/`fit_streaming`, `memmap2`
  backing and persistence/export (Wave 7), pruning, and the ensemble estimators.
- Companion tests: hand-checkable splits, impurity calculations, dense/sparse equivalence,
  class weights, feature subsampling, determinism, depth/leaf constraints, large data,
  concurrent prediction and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `decision-trees`: CART classifier and regressor — builder surface, impurity and split
  search, tree structure, prediction and probability estimates, feature importances,
  dense/sparse representation parity, determinism and validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_tree` workspace member; root `Cargo.toml` membership.
- **APIs:** additive. `SKDecisionTreeClassifier`, `SKDecisionTreeRegressor`, their builders
  and fitted models.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `ndarray`, `sprs`, `rayon`;
  `sciencekit_metrics` as a dev-dependency.
- **Downstream:** unblocks `SKRandomForest`, `SKBagging`, `SKGradientBoosting`,
  `SKHistGradientBoosting` (Wave 4) and `SKIsolationForest` (Wave 5).
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Persistence,
  memory-mapped backing and model export are deferred to Wave 7 and recorded in the PRD.
