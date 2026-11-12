# Proposal

## Why

A single CART tree is high-variance: a small change in the training sample can change the
root split and the whole structure. Breiman's random forest averages many de-correlated
trees and estimates the generalization error from out-of-bag samples. It is the canonical
Wave 4 ensemble and the first consumer of the tree crate's structure and feature
importances.

## What Changes

- New sub-crate `sciencekit_ensemble` with `SKRandomForestClassifier<F>` and
  `SKRandomForestRegressor<F>`, each with a distinct fitted model.
- Builder-only surface: `number_of_trees` (default 100), `criterion`, `maximum_depth`,
  `minimum_samples_split`, `minimum_samples_leaf`, `maximum_features` (`Sqrt` default for
  forests), `bootstrap` (default true), `maximum_samples` (default all), `random_seed`
  (default 0), `class_weight` (classifier), `oob_score` (default false), `backend`
  (default `Faer`) and `execution_mode` (default `Automatic`).
- Fit builds one CART tree per bootstrap sample in parallel with rayon. Each tree gets a
  deterministic seed derived from `random_seed` and its index, so the forest is identical
  regardless of thread count.
- Lock-free aggregation: each tree writes its own slot; no shared mutable accumulator.
- Out-of-bag estimation: every tree records the rows not drawn in its bootstrap sample,
  and the model exposes the OOB score (accuracy for classification, R² for regression)
  when enabled.
- Prediction averages the tree class probabilities (classifier) or responses
  (regressor), with deterministic class ordering and tie-breaking.
- Model accessors: tree count, per-tree feature importances aggregated into a normalized
  forest importance, and the OOB score.
- Dense and sparse representation parity through the tree estimators; sparse input is
  never densified.
- Validation: empty input, zero tree count, invalid sample/depth/split/leaf parameters,
  feature-count mismatch, non-finite values, single-class classification and an invalid
  OOB configuration.
- Explicit non-goals: missing-value routing, extremely randomized trees, streaming/
  `partial_fit`, `memmap2` backing and persistence/export (Wave 7).
- Companion tests: hand-checkable forests, bootstrap determinism, OOB correctness,
  dense/sparse equivalence, concurrency, large data and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `random-forests`: `SKRandomForestClassifier` and `SKRandomForestRegressor` — builder
  surface, bootstrap and feature-subsampling contract, parallel tree fitting, OOB
  estimation, aggregation, representation parity, determinism and validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_ensemble` workspace member; root `Cargo.toml`
  membership.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `sciencekit_tree`, `ndarray`,
  `sprs`, `rayon`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** gives Wave 4 its first ensemble and a baseline for the boosting changes;
  the OOB machinery is reused by `SKBagging` (W4.3).
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Model export is
  deferred to Wave 7 and recorded in the PRD.
