# Proposal

## Why

Random forests are specialized tree ensembles. Bagging is the generic variance-reduction
meta-estimator: it fits any supervised estimator on bootstrap samples and averages its
predictions. It gives the library a reusable ensemble surface for models that are not
trees, and it exercises the estimator contracts generically rather than through a
tree-specific shortcut.

## What Changes

- `SKBaggingClassifier<B>` and `SKBaggingRegressor<B>` in `sciencekit_ensemble`, generic
  over a base estimator configuration `B`.
- Builder-only surface: `number_of_estimators` (default 10), `maximum_samples` (default
  all, or a count/fraction), `maximum_features` (default all, or a count/fraction),
  `bootstrap` (default true), `bootstrap_features` (default false), `oob_score` (default
  false), `random_seed` (default 0), `backend` (default `Faer`) and `execution_mode`
  (default `Automatic`).
- Generic fitting contract: `B` is the configured base estimator; bounds require
  `B: Clone + Send + Sync + SKSupervisedFit<F>`, and the base model must implement
  `SKClassifierPredictor<F>` (classifier) or `SKRegressorPredictor<F>` (regressor). The
  bagging model stores `Vec<B::Model>` and the feature subset used by each estimator.
- Bootstrap rows and, optionally, features; per-estimator deterministic seeds derived from
  `random_seed` and the estimator index; parallel fitting with rayon and lock-free slots.
- Prediction routes each estimator through its feature subset, averages class
  probabilities (classifier) or responses (regressor), and ties break by the smallest
  canonical class index.
- OOB estimation reusing the shared bootstrap masks: accuracy for classification and R²
  for regression when enabled.
- Model accessors: estimator count, per-estimator feature subsets, OOB score.
- Dense and sparse representation parity at the bagging level: the base estimator receives
  the same representation it supports; a representation the base estimator rejects is
  reported with its own error.
- Validation: empty input, zero estimator count, invalid sample/feature fractions, a base
  estimator that does not fit the supervision axis, feature-count mismatch and non-finite
  values.
- Explicit non-goals: heterogeneous base estimators (a runtime trait object), stacking/
  voting, streaming/`partial_fit`, `memmap2` backing and export (Wave 7).
- Companion tests: hand-checkable bagging, bootstrap determinism, feature-subset routing,
  OOB correctness, dense/sparse equivalence with a sparse-capable base estimator,
  concurrency, large data and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `bagging-ensembles`: `SKBaggingClassifier`/`SKBaggingRegressor` — generic base-estimator
  contract, bootstrap and feature subsampling, parallel fitting, prediction aggregation,
  OOB estimation, determinism and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_ensemble` gains the bagging modules; no new crate.
- **APIs:** additive. The generic estimator types are the first public ensemble parameters
  over another estimator.
- **Dependencies:** `sciencekit_common`, `ndarray`, `sprs`, `rayon`; `sciencekit_metrics`
  as a dev-dependency.
- **Downstream:** gives model selection a generic variance-reduction wrapper and validates
  the fit/predict contracts as a reusable composition surface.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
