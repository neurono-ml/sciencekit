# Proposal

## Why

Random forests average independent trees; gradient boosting fits an additive model stage
by stage, each tree correcting the current model's negative gradient. It is the strongest
classical tabular baseline and the direct predecessor of the histogram variant. The tree
crate already supplies the structure, constraints and split search; the new work is the
loss/gradient contract, Newton leaf values, shrinkage and early stopping.

## What Changes

- `SKGradientBoostingClassifier<F>` and `SKGradientBoostingRegressor<F>` in
  `sciencekit_ensemble`, sharing a boosting substrate.
- Builder-only surface: `number_of_estimators` (default 100), `learning_rate` (default
  0.1), `loss` (regression `SquaredError` default, `AbsoluteError`, `Huber`;
  classification `LogLoss`), `maximum_depth` (default 3), `minimum_samples_leaf` (default
  1), `maximum_features` (default all), `subsample` (default 1.0), `l2_regularization`
  (default 0.0), `random_seed` (default 0), `validation_fraction` (default 0.1),
  `n_iter_no_change` (default none), `tolerance` (default 1e-4), `backend` (default
  `Faer`) and `execution_mode` (default `Automatic`).
- Loss contract: each loss computes the initial prediction, the negative gradient and the
  hessian per row. Classification uses the log-loss with a sigmoid (binary) or softmax
  (multiclass) link; regression uses the configured squared/absolute/Huber loss.
- Stage fitting: a regression tree is grown on the negative gradient (and Hessian), then
  each leaf value is replaced by the Newton step `-Σg / (Σh + λ)` and scaled by the
  learning rate. The `sciencekit_tree` crate gains the structure-only build and leaf-value
  replacement surface needed for this.
- Multinomial classification fits one tree per class per stage; binary fits one tree per
  stage.
- Optional stochastic boosting (`subsample < 1`) and feature subsampling use the seeded
  generator.
- Early stopping on an internal validation split when `n_iter_no_change` is set; the model
  reports the number of stages actually fitted.
- Model accessors: initial prediction, per-stage trees, learning rate, feature importances
  and the fitted stage count.
- Dense and sparse representation parity through the tree base learner; sparse input is
  never densified.
- Validation: empty input, non-positive learning rate or estimator count, invalid
  subsample/feature/regularization parameters, single-class classification, non-finite
  values and an invalid validation fraction.
- Explicit non-goals: histogram binning (W4.5), categorical features, missing-value
  routing, streaming/`partial_fit`, `memmap2` backing and export (Wave 7).
- Companion tests: hand-computed gradients and Newton leaves, binary/multiclass
  classification, regression losses, shrinkage, subsampling determinism, early stopping,
  dense/sparse equivalence, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `gradient-boosting`: the loss/gradient/hessian contract, stage-wise additive fitting,
  Newton leaf values, shrinkage, stochastic subsampling, early stopping and the
  classifier/regressor prediction surface.

### Modified Capabilities

- `decision-trees`: adds the structure-only build and leaf-value replacement surface needed
  by boosting, without changing the public CART estimator behavior.

## Impact

- **Code:** `crates/sciencekit_ensemble` gains the boosting modules; `sciencekit_tree`
  gains an internal/advanced API for structure building and leaf replacement.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `sciencekit_tree`, `ndarray`,
  `sprs`, `rayon`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** unblocks `SKHistGradientBoosting` (W4.5), which reuses the loss and
  boosting substrate with histogram splits.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
