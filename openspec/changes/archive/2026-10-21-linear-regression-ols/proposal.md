## Why

Wave 1 closed with the preprocessing and impute estimators; Wave 2 (PRD §12 Phase 2,
`sciencekit_linear_model`) opens with `SKLinearRegression` — the first *supervised*
estimator and the first consumer of the backend's `lstsq` kernel. Every supervised
algorithm after it (ridge, lasso, logistic) copies this change's estimator/model shape,
its supervised-fit wiring, and its honest treatment of the streaming reformulation of a
dense linear system.

Doing ordinary least squares first is deliberate: the dense math is a single SVD
least-squares solve, so any incoherence in the supervised fit contract, the design-matrix
convention, the backend dispatch or the Gram-based streaming path surfaces here — before
it is copied into a dozen regressors and classifiers.

## What Changes

- **New workspace crate `crates/sciencekit_linear_model`** (PRD §3.1: one crate per change)
  holding `SKLinearRegression`, split into an immutable, scalar-free estimator
  (configuration) and a fitted `SKLinearRegressionModel<F>` (learned state and export
  boundary), depending on `sciencekit_common` and `sciencekit_math`.
- **Builder surface:** `SKLinearRegression::build()` exposes `include_intercept(bool)`
  (default `true`), `backend(SKBackendKind)` (default `Faer`) and
  `execution_mode(SKExecutionMode)` (default `Automatic`); the direct constructor stays
  private. The estimator carries no scalar in its configuration, so it is **not** generic
  over `F`; the fitted model is.
- **Dense `fit` (SVD least squares):** the design matrix `A = [1 ‖ X]` (unless
  `include_intercept` is false) is solved through the backend's `lstsq` (the pseudoinverse
  path) under a `Fit` observability span; `rank` and the singular values are surfaced on
  the model rather than hidden. Normal equations (`solve(AᵀA, Aᵀy)`) are rejected — they
  square the condition number.
- **`SKRegressorPredictor`:** the fitted model predicts responses as `X·coefficients +
  intercept`, returning `Array1<F>` in the model scalar.
- **Incremental `partial_fit` and out-of-core `fit_streaming`:** both accumulate the Gram
  matrix `AᵀA` and right-hand side `Aᵀy` per batch and solve once at the end, so the two
  regimes and the dense fit agree; the streaming reformulation's precision trade-off (the
  squared condition number) is documented, not hidden.
- **Kernels `sk_gram_matrix` and `sk_gram_vector`** in `sciencekit_math`: the `XᵀX` and
  `Xᵀy` cross-product reductions the Gram accumulation needs, with the same sequential /
  race-free parallel contract as the existing axis reductions.
- **Supervised streaming contract** in `sciencekit_common`: `SKSupervisedBatch` (owned
  features + targets), the `SKSupervisedLazySource` fallible iteration, and
  `sk_run_supervised_streaming_driver`, which reuses the existing prefetch pipeline. The
  existing `SKDataBatch` carries features only, so a supervised out-of-core fit needs this
  pair; the feature-only driver is unchanged.
- **Companion test modules** (`*_tests.rs`) covering dense, incremental and streaming
  fits, f32/f64, intercept on/off, rank-deficient designs, concurrency and the model
  export boundary.

## Capabilities

### New Capabilities

- `linear-regression`: the `SKLinearRegression` estimator/model contract — builder surface
  and defaults, the design-matrix and intercept convention, the SVD least-squares fit and
  its surfaced rank/singular values, the `SKRegressorPredictor` response, the Gram-based
  incremental and streaming fit equivalence, and the error taxonomy.

### Modified Capabilities

- `higher-order-kernels`: adds the cross-product reductions `sk_gram_matrix` (`XᵀX`) and
  `sk_gram_vector` (`Xᵀy`), the matrix/vector companions of the axis reductions, with the
  same parallelism and grain contract.
- `streaming-batches`: adds the supervised streaming contract (`SKSupervisedBatch` /
  `SKSupervisedLazySource`) and the prefetch driver entry point that serves both the
  feature-only and the supervised sequential sources — discovered during implementation,
  because the existing `SKDataBatch` carries only features and a supervised out-of-core fit
  needs the targets alongside them.

## Impact

- **New code:** `crates/sciencekit_linear_model/` (`linear_regression/{builder,
  core_implementation,model,streaming}` + companion `*_tests.rs`) and one new
  `[workspace] members` entry in the root `Cargo.toml`.
- **`sciencekit_math`:** `kernels/` gains `sk_gram_matrix` and `sk_gram_vector`; the crate
  root re-exports them.
- **`sciencekit_common`:** `batching/` gains `SKSupervisedBatch` / `SKSupervisedLazySource`
  and `execution/streaming_driver.rs` gains `sk_run_supervised_streaming_driver`, sharing
  the existing prefetch engine with the unchanged `sk_run_streaming_driver`.
- **`sciencekit_math`:** `backend/faer_backend.rs` switches the least-squares kernel to the
  thin SVD (a discovered performance bug fix — see `design.md`).
- **Workspace:** a `[profile.dev.package."*"] opt-level = 2` block keeps dependency code
  optimized in dev/test builds.
- **Dependencies:** none new beyond the existing `ndarray`/`rayon`; the streaming path
  reuses the Phase 0 streaming executor and the backend's `lstsq`.
- **Public API:** new exports `SKLinearRegression`, `SKLinearRegressionBuilder`,
  `SKLinearRegressionModel` — all additive, no existing item changes.
- **Downstream:** `ridge-regression`, `lasso-elasticnet-coordinate-descent` and the rest of
  Wave 2 follow this crate/estimator/model layout and its supervised-fit wiring;
  `docs/src/algorithms.md` "Linear models · `sciencekit_linear_model`" stops being a
  placeholder; a new Developer Book chapter and API pages document the estimator.
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large data,
  streaming ≡ dense and incremental ≡ dense within float tolerance, safe under concurrency,
  the model carries its backend kind and is the export boundary. Model export
  (Safetensors/JSON) and the metrics suite are deferred to their Phase 7 / W6.3 changes,
  as the scaler changes already do.
