## Why

Wave 2 opened with `SKLinearRegression` (W2.1) — an *unregularized* estimator whose solve
is a single least-squares call. `SKRidge` (W2.2) is the first **regularized** linear model:
its objective is `argmin ‖y − Xβ‖₂² + α‖β‖₂²`, so the normal equations gain an `αI` term
and the estimator must choose between solvers with different numerical properties, and must
serve two input representations. The wave plan (`wave-plan-foundation`, task 4) scopes W2.2
as *"`SKRidge` (Cholesky dense, sparse_cg, SVD fallback, zero-copy sparse `LinearOperator`)"*,
and deliberately makes ridge the entry point for **sparse input** — `SKLinearRegression`
rejects it, while `SKDataView` has carried the `Sparse(CsMatView)` variant since Wave 1.
Getting the solver portfolio, the intercept convention and the multi-`alpha` model shape
right here means `RidgeCV` (W6), Lasso/ElasticNet (W2.3) and the sparse linear models that
follow inherit one coherent, statistically correct regularized-regression contract.

## What Changes

- **New `ridge` module in `crates/sciencekit_linear_model`** holding `SKRidge<F>`, split into
  an immutable builder-configured estimator (configuration carries `alpha`, so the estimator
  is generic over `F`) and a distinct fitted `SKRidgeModel<F>` (learned state and export
  boundary).
- **Builder surface:** `SKRidge::<F>::build()` exposes `alpha(...)` (a scalar `F` or a
  sequence — `Vec<F>`/`Array1<F>`/`[F; N]` — defaulting to `1.0`), `include_intercept(bool)`
  (default `true`), `solver(SKRidgeSolver)` (default `Automatic`), `tolerance(F)` and
  `maximum_number_of_iterations(usize)` (for the iterative solver), `backend(SKBackendKind)`
  (default `Faer`) and `execution_mode(SKExecutionMode)` (default `Automatic`); the direct
  constructor stays private.
- **Solver portfolio** (`SKRidgeSolver`):
  - `Cholesky` — the centered Gram `Ĉ = XᵀX − n·x̄x̄ᵀ` plus `αI`, solved through the
    backend's Cholesky; works for dense **and** sparse input (the Gram is `p × p` dense).
  - `Svd` — dense only; the centered design's SVD decomposes once and each `alpha` is applied
    as `β = V diag(σ/(σ²+α)) Uᵀ ỹ`; the robust path for `α = 0` or rank-deficient designs;
    rejects sparse input with a structured error pointing at `Cholesky`/`SparseCg`.
  - `SparseCg` — minimizes `(DᵀD + αP)β = Dᵀy` (`D = [1 ‖ X]`, `P = diag(0, 1, …, 1)`) by
    **conjugate gradient**, applying `DᵀD` through a **zero-copy `SKLinearOperator` over the
    borrowed `CsMatView`** — `AᵀA` is never materialized; honors `tolerance`/
    `maximum_number_of_iterations` and raises `SKError::NotConverged` when exhausted.
  - `Automatic` — dense resolves to `Cholesky` (falling back to `Svd` when `α = 0` or the
    Cholesky factor fails), sparse resolves to `SparseCg`.
- **Unpenalized intercept:** the intercept is never shrunk by the penalty. The Gram/SVD
  solvers center the data and recover `β₀ = ȳ − x̄ᵀβ`; the CG solver uses the algebraically
  equivalent unpenalized augmentation (`P = diag(0, 1, …, 1)`). `include_intercept(false)`
  drops the intercept.
- **Multi-`alpha` model:** one fit yields one solution per `alpha`; the model exposes
  `coefficients()` (`n_alphas × n_features`), `intercepts()` (`n_alphas`) and `alphas()`.
  Regressor `predict` is defined for the single-`alpha` configuration; with several alphas it
  returns a structured error directing the consumer to `predict_with_alpha_index`.
- **`SKRegressorPredictor<F>`:** the fitted model predicts responses as `X·β + β₀` for the
  selected `alpha`.
- **Incremental `partial_fit` and out-of-core `fit_streaming`:** both accumulate the centered
  Gram state (`XᵀX`, `Xᵀy`, `Σx`, `Σy`, row count) per batch and solve the accumulated normal
  equations once at the end, so the two regimes and the dense `fit` agree; a `SparseCg` intent
  met at streaming falls back to the Gram solve and records the solver actually used.
- **New `solvers` module in `sciencekit_math`:** the `SKLinearOperator<F>` trait (zero-copy
  SPD operator surface) and `sk_conjugate_gradient` (SPD conjugate gradient with a structured
  non-convergence error), plus a zero-copy CSR normal-equation operator — the substrate the
  sparse ridge path consumes and later linear models and `SKTruncatedSVD` will reuse.
- **Companion test modules** (`*_tests.rs`) covering dense/sparse fits, all solvers,
  `alpha` scalar and array, intercept on/off, rank deficiency, incremental and streaming
  equivalence, `f32`/`f64`, non-convergence, concurrency and the model export boundary.

## Capabilities

### New Capabilities

- `ridge-regression`: the `SKRidge` estimator/model contract — builder surface and defaults,
  the `SKRidgeSolver` portfolio and its `Automatic` resolution, the unpenalized-intercept
  convention, the multi-`alpha` model shape and per-`alpha` prediction, the centered Gram
  incremental/streaming equivalence, and the error taxonomy.
- `iterative-linear-solvers`: the `sciencekit_math` substrate — the `SKLinearOperator<F>`
  zero-copy operator trait, the `sk_conjugate_gradient` SPD solve and its convergence/
  non-convergence contract, and the zero-copy CSR normal-equation operator.

### Modified Capabilities

_None._ The ridge work adds new capabilities; it does not change the requirements of the
existing `higher-order-kernels`, `dense-sparse-products`, `linear-regression`,
`backend-selection` or `streaming-batches` capabilities (the CG substrate is cohesive enough
to own its own capability rather than extend the reduction family).

## Impact

- **New code:** `crates/sciencekit_linear_model/src/ridge/` (`builder`, `core_implementation`,
  `model/`, `solving/`, `sparse/`, `gram` + companion `*_tests.rs`; ridge carries its own
  centered Gram accumulator, leaving the W2.1 one untouched).
- **`sciencekit_math`:** new `solvers/` module (`SKLinearOperator`, `sk_conjugate_gradient`,
  the CSR normal-equation operator); the crate root re-exports them.
- **Dependencies:** `sprs 0.11` promoted from a `dev-dependency` to a production dependency of
  `sciencekit_linear_model`; no new crate beyond the already-pinned `ndarray`/`rayon`/`faer`/
  `sprs`.
- **Public API:** new exports `SKRidge`, `SKRidgeBuilder`, `SKRidgeModel`, `SKRidgeSolver` and
  the `sciencekit_math` solver items — all additive, no existing item changes.
- **Downstream:** `RidgeCV` (W6) consumes the per-`alpha` model shape; Lasso/ElasticNet
  (W2.3), logistic regression (W2.4) and the sparse linear models reuse the `SKLinearOperator`
  and the centered-Gram accumulator; `docs/src/algorithms.md` gains the `SKRidge` entry, and a
  new Developer Book chapter documents the estimator.
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large data; dense,
  incremental and streaming fits agree within float tolerance; the three solvers agree within
  a documented relative tolerance; safe under concurrency; the model carries its backend kind
  and is the export boundary. Model export (Safetensors/JSON) and the metrics suite are
  deferred to their Phase 7 / W6.3 changes, as the scaler and W2.1 changes already do.
