# Proposal

## Why

Exhaustive `SKGridSearchCV` evaluates every candidate at the full data size, which is
wasteful when most candidates are clearly worse. Successive halving evaluates all candidates
on a small resource, keeps the best fraction, increases the resource and repeats. It is the
second standard search strategy and the wave-plan companion to grid search.

## What Changes

- `SKHalvingGridSearchClassifier<E>` and `SKHalvingGridSearchRegressor<E>` in
  `sciencekit_model_selection`.
- Builder-only surface: candidate list, splitter, scoring policy, `factor` (default 3),
  `minimum_resource` (default 1), `maximum_resource` (default all samples), `resource`
  (`NumberOfSamples` default), `aggressive_elimination` (default false), `random_seed`
  (default 0), `refit` (default true), `error_score`, `execution_mode`.
- Successive-halving loop: evaluate all surviving candidates on the current resource,
  keep the top `ceil(n / factor)`, increase the resource by `factor`, and repeat until one
  candidate or the maximum resource is reached.
- Deterministic candidate elimination (ties by candidate index) and seeded subsampling of
  the resource budget.
- Reuses the grid-search scoring and refit machinery; the final model exposes the
  elimination trace, best candidate/score and the refit model.
- Dense and sparse inputs through the shared gathering helper; no densification.
- Validation: factor < 2, invalid resource bounds, empty candidates, incompatible scorer,
  candidate failure under `Raise` and prediction without refit.
- Explicit non-goals: Bayesian/randomized search, multi-resource search, `memmap2` and
  export.
- Companion tests: hand-checkable elimination trace, deterministic seeds, resource growth,
  aggressive elimination, dense/sparse, refit and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `halving-grid-search`: successive-halving evaluation, resource scheduling, candidate
  elimination, refit and its validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_model_selection` gains the halving modules.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_metrics`, `ndarray`, `sprs`, `rayon`.
- **Downstream:** gives users a cheaper hyperparameter search for large grids.
- **Acceptance criteria:** small and large data, concurrent folds, deterministic
  elimination, companion `*_tests.rs` modules. Export is deferred to Wave 7.
