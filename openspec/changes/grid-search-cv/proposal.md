# Proposal

## Why

`model-selection-core` supplies splits and scores, but there is no way to search a set of
candidate estimator configurations, aggregate fold scores and refit the winner. `SKGridSearchCV`
is the standard hyperparameter search surface and the first meta-estimator in the library.

## What Changes

- New `SKGridSearchCVClassifier<E>` and `SKGridSearchCVRegressor<E>` in
  `sciencekit_model_selection`, generic over the configured base estimator `E`.
- Builder-only surface: a candidate list (`Vec<E>`), a splitter (`SKKFold`,
  `SKStratifiedKFold` or `SKGroupKFold`), a scoring policy, `refit` (default true),
  `error_score` (`Raise` default or a numeric value), `execution_mode` (default
  `Automatic`) and an optional group vector.
- Fit evaluates every candidate on every fold in parallel, aggregates mean and standard
  deviation per candidate, selects the highest mean score (ties by candidate index) and,
  when `refit` is true, refits the winner on the full data.
- The fitted model exposes `best_candidate_index`, `best_score`, `results()`, the best
  fitted model and the scoring policy; classifier/regressor wrappers implement the matching
  predictor and delegate to the best model.
- Cross-validation uses the index splits and row-gathering helper from
  `model-selection-core`; dense and sparse inputs are gathered without densification.
- Scoring policies: classification accuracy/F1/precision/recall/AUC/log loss/balanced
  accuracy; regression negative MSE/MAE/R². Custom scoring is a documented extension point.
- Validation: empty candidate list, invalid splitter/group inputs, a scorer incompatible
  with the estimator family, a candidate failure under `error_score = Raise` and
  `refit = false` prediction.
- Explicit non-goals: halving/randomized search (W6.5), nested CV, stacking, `memmap2` and
  export.
- Companion tests: hand-checkable candidate/fold grids, deterministic tie-breaking,
  parallel-equals-sequential, dense/sparse, refit behavior, error scores and metric
  production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `grid-search`: candidate evaluation across folds, score aggregation, best-candidate
  selection, refit, classifier/regressor wrappers and error scoring.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_model_selection` gains the grid-search modules; no new
  crate.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_metrics`, `ndarray`, `sprs`, `rayon`.
- **Downstream:** gives users hyperparameter selection and exercises the split/scoring
  foundation.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent folds,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7.
