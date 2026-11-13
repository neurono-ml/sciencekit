# Proposal

## Why

PCA centers the data, which is exactly what a sparse text-like matrix cannot afford.
TruncatedSVD computes a low-rank factorization of the uncentered matrix, keeps the input
sparse and is the standard path for latent semantic analysis and sparse dimensionality
reduction. It reuses the randomized SVD kernel from this wave.

## What Changes

- `SKTruncatedSVD<F>` in `sciencekit_decomposition`, with a distinct fitted model.
- Builder-only surface: `number_of_components` (default 2), `algorithm` (`Randomized`
  default), `number_of_iterations` (default 5), `tolerance` (default 0.0),
  `random_seed` (default 0), `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`).
- Fit runs the randomized SVD on the uncentered design, accepts dense and sparse views,
  dispatches once and never densifies a sparse matrix.
- The model exposes components, singular values, explained variance, explained-variance
  ratio and the fitted number of components; `transform` and `inverse_transform` are
  available.
- Validation: empty input, zero or excessive components, non-positive iterations/tolerance,
  an unsupported algorithm and feature-count mismatch.
- Explicit non-goals: centering, ARPACK/IRLBA, missing values, GPU, `memmap2` and export
  (Wave 7).
- Companion tests: hand-checkable rank-1/rank-2 matrices, reconstruction, variance ratio,
  dense/sparse equivalence, no densification, deterministic signs, concurrency, large data
  and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `truncated-svd`: `SKTruncatedSVD` — randomized low-rank factorization, component/variance
  state, transform/inverse transform, dense/sparse representation parity and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_decomposition` gains the truncated-SVD module and model.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math` (randomized SVD), `ndarray`,
  `sprs`, `rayon`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** provides the sparse alternative to PCA and the low-rank stage for later
  text pipelines.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent transform,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
