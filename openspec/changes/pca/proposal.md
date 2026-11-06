# Proposal

## Why

PCA is the canonical linear dimensionality-reduction estimator and the numeric base for
whitening, visualization and downstream models. The backend already exposes full SVD and
symmetric eigh, and the randomized SVD kernel (this wave) supplies the large-data path.
PCA is also the change that must state clearly why centering makes its sparse path a
documented dense-only exception.

## What Changes

- New sub-crate `sciencekit_decomposition` with `SKPCA<F>` and a distinct fitted model.
- Builder-only surface: `number_of_components` (default `None`, meaning
  `min(n_samples, n_features)`), `svd_solver` (`Auto` default, `Full`, `Randomized`,
  `CovarianceEigh`), `whiten` (default false), `tolerance` (default 0.0),
  `number_of_power_iterations` (default 5), `random_seed` (default 0), `backend` (default
  `Faer`) and `execution_mode` (default `Automatic`).
- Fit computes the feature means, centers the design, runs the selected solver and stores
  components, explained variance, explained-variance ratio, singular values and the mean.
- `transform` and `inverse_transform` are provided; `fit_transform` is available; `whiten`
  scales the transformed components by the inverse singular values.
- Dense input only: centering a sparse matrix destroys its sparsity, so a sparse view is
  rejected with `UnsupportedRepresentation` and a suggestion to use `SKTruncatedSVD` (the
  documented dense-only exception). The decomposition family still covers sparse data
  through TruncatedSVD and IncrementalPCA.
- Solver resolution: `Auto` chooses full SVD for small designs and randomized SVD for
  large ones; `CovarianceEigh` uses the covariance matrix and is appropriate when
  `n_samples > n_features`.
- Validation: empty input, zero or excessive components, non-positive tolerance/power
  iterations, invalid solver and feature-count mismatch.
- Explicit non-goals: sparse PCA, kernel PCA, incremental PCA (W5.6), `memmap2` and
  export (Wave 7).
- Companion tests: hand-checkable 2-D projection, reconstruction, explained variance,
  white transform, solver equivalence, determinism, concurrency, large data and metric
  production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `principal-component-analysis`: `SKPCA` — builder surface, solver portfolio, centering,
  component/variance state, transform/inverse transform, whitening, the dense-only
  exception and validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_decomposition` workspace member; root `Cargo.toml`
  membership.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `ndarray`, `sprs`, `rayon`;
  `sciencekit_metrics` as a dev-dependency.
- **Downstream:** unblocks TruncatedSVD and IncrementalPCA in the same crate, and gives
  pipelines a reduction/whitening stage.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent transform,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
