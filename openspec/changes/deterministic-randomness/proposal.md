# Proposal

## Why

Stochastic algorithms are arriving in every wave — SGD, k-means++, MiniBatchKMeans, random
forests, bagging, boosting, randomized SVD, probability calibration and isolation forests.
Today the only seeded generator is a `pub(crate)` SplitMix64 inside
`sciencekit_linear_model`; every new crate would either duplicate it or reach for an
ambient RNG. Reproducibility should be a library rule, not a per-change accident.

## What Changes

- Promote the seeded SplitMix64 generator to `sciencekit_common` as `SKRandomGenerator`,
  with `next_u64`, `next_f64`, `uniform`, `normal`, `shuffled_indices` and
  `sample_indices_without_replacement`.
- Add `sk_derive_seed(global_seed, domain, index)` so a component (tree, initialization,
  stage, fold) gets a deterministic sub-seed independent of thread scheduling.
- Add the cross-cutting rule: every stochastic estimator builder exposes
  `random_seed(u64)` defaulting to `0`, derives all sub-seeds from it, never uses an
  ambient RNG, and collects results in a stable order.
- Refactor `sciencekit_linear_model`'s SGD code to consume the shared generator; no
  behavioral change for a fixed seed.
- Add golden-vector tests for the generator and reproducibility tests for the estimators
  that will consume it.
- No new external dependency.

## Capabilities

### New Capabilities

- `deterministic-randomness`: the seeded generator contract, sub-seed derivation, the
  `random_seed` builder rule, deterministic collection and the reproducibility scope.

### Modified Capabilities

None.

## Impact

- **Code:** `sciencekit_common::random` (new module and re-exports);
  `sciencekit_linear_model` drops its private generator and imports the shared one.
- **APIs:** additive for common; internal for linear_model. No public estimator signature
  changes beyond the already-planned `random_seed` setters.
- **Downstream:** Wave 3–5 changes stop duplicating RNG code; every stochastic estimator
  uses one seed convention.
- **Acceptance criteria:** golden vectors, thread-count invariance, same-seed
  reproducibility and a documented cross-platform tolerance.
