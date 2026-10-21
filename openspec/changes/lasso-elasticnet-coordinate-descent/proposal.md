## Why

Wave 2 opened with `SKLinearRegression` (W2.1, unregularized) and `SKRidge` (W2.2, an L2
penalty whose normal equations are a single smooth linear solve). `SKRidge` deliberately
left out the models whose penalty is **non-differentiable at the origin**: the L1 term makes
the solution have no closed form, promotes **exact zeros** (feature selection) and is *not*
a function of the Gram matrix `XᵀX` alone. The wave plan (`wave-plan-foundation`, task 4)
scopes W2.3 as *"`SKLasso` + `SKElasticNet` (coordinate descent + gap-safe screening, rayon
across multi-target)"*, anchored on Tibshirani 1996, Zou & Hastie 2005 and Friedman,
Hastie & Tibshirani 2010 (glmnet). Lasso/ElasticNet are the entry point for **sparse design
input as a first-class citizen** — a sparse design must stay sparse through the solve —
and they establish the **coordinate-descent substrate** the later L1-regularized models
(logistic regression, linear SVM) and `RidgeCV`-style paths reuse.

## What Changes

- **New `coordinate_descent` module in `crates/sciencekit_linear_model`** holding the
  shared, representation-agnostic solver substrate: the objective configuration, the
  hyperparameter validation, the alpha-conversion trait, the cyclic/random sweep
  selection, the coordinate-descent iteration (soft-threshold updates with an unpenalized
  centered intercept), the gap-safe screening with a KKT reactivation guarantee, and the
  retained-design accumulator that backs the incremental and streaming regimes.
- **New `lasso` module** holding `SKLasso<F>` (builder-configured estimator) and the
  distinct fitted `SKLassoModel<F>` (learned state and export boundary).
- **New `elastic_net` module** holding `SKElasticNet<F>` and the distinct fitted
  `SKElasticNetModel<F>`; it shares the whole coordinate-descent substrate with lasso and
  adds the `l1_ratio` mixing parameter.
- **Builder surface (both estimators):** `alpha(...)` (a scalar `F` or a sequence —
  `Vec<F>`/`Array1<F>`/`[F; N]`/`&[F]` — defaulting to `1.0`), `include_intercept(bool)`
  (default `true`), `tolerance(F)` (default `1e-4`), `maximum_number_of_iterations(usize)`
  (default `1000`), `positive(bool)` (default `false`), `selection(SKSelection)` (default
  `Cyclic`), `screening(bool)` (default `true`), `backend(SKBackendKind)` (default `Faer`)
  and `execution_mode(SKExecutionMode)` (default `Automatic`); ElasticNet adds
  `l1_ratio(F)` (default `0.5`, validated to `[0, 1]`). The direct constructor stays
  private.
- **Coordinate-descent solver (glmnet-style):** each coordinate is updated by a
  soft-threshold of the partial residual gradient, `w_j = S(x_jᵀr / n + a_j w_j, α·ρ) /
  (a_j + α(1−ρ))` with `a_j = ‖x_j‖² / n`; the intercept is an unpenalized coordinate
  updated to the residual mean each sweep; convergence is the maximum coordinate change
  against `tolerance`, bounded by `maximum_number_of_iterations` and surfaced as
  `SKError::NotConverged` when exhausted.
- **Gap-safe screening:** a strong initial active set from the gradient at the origin,
  a **gap-safe safe-screening** elimination derived from the primal–dual gap bound for the
  pure-L1 (lasso) objective, and a **KKT reactivation pass** that adds back any coordinate
  whose full-gradient KKT condition is violated — so the screened solution provably equals
  the unscreened solution. The fitted model exposes how many coordinates were screened.
- **Dense and CSR sparse input:** the solver runs over both without densifying a sparse
  design, sharing one coordinate loop through a column-access abstraction. Public APIs take
  zero-copy `ArrayView2`/`CsMatView`.
- **Unpenalized intercept by centering:** the intercept is never shrunk; it is recovered as
  the residual mean and the feature updates run on the mean-zero residual, algebraically
  equivalent to centering the design (which would densify a CSR matrix).
- **Incremental `partial_fit` and out-of-core `fit_streaming`:** because the L1 objective
  is not a function of the Gram matrix, the two regimes **retain the design** (dense rows or
  stacked CSR batches) in a shared accumulator and re-solve at read time; the memory
  trade-off is documented. Batches of any size or order converge to the dense fit.
- **Distinct fitted models:** `coefficients()` (`n_alphas × n_features`), `intercepts()`
  (`n_alphas`), `alphas()`, `l1_ratio()` (ElasticNet), `backend()`, `include_intercept()`,
  `positive()`, `row_count()`, `screened_count()`, per-alpha accessors and the
  `SKRegressorPredictor<F>` impl (single-alpha plain `predict`, explicit
  `predict_with_alpha_index`).
- **Rayon across independent solves:** with several alphas configured, the independent
  per-alpha coordinate-descent solves run in parallel over the compute pool (the wave
  plan's parallel axis), with results independent of the thread count.
- **Companion test modules** (`*_tests.rs`) covering dense/sparse fits, both estimators,
  the closed-form one-feature and orthogonal solutions, KKT satisfaction, screening
  equivalence, alpha scalar and sequence, intercept on/off, `positive`, `selection`,
  incremental and streaming equivalence, `f32`/`f64`, concurrency and the model boundary.

## Capabilities

### New Capabilities

- `coordinate-descent-solver`: the shared `sciencekit_linear_model` substrate — the
  objective configuration and validation, the alpha-conversion trait, the cyclic/random
  selection, the soft-threshold coordinate iteration, the unpenalized centered intercept,
  the gap-safe screening with KKT reactivation, the dense/sparse shared coordinate loop and
  the retained-design accumulator.
- `lasso`: the `SKLasso` estimator/model contract — builder surface and defaults, the L1
  penalty semantics, prediction, the incremental/streaming regimes and the error taxonomy.
- `elastic-net`: the `SKElasticNet` estimator/model contract — builder surface and
  defaults including `l1_ratio`, the mixed L1/L2 penalty semantics (degenerating to lasso
  at `l1_ratio = 1`), prediction, the incremental/streaming regimes and the error taxonomy.

### Modified Capabilities

_None._ The change adds new capabilities; it does not change the requirements of
`ridge-regression`, `linear-regression`, `backend-selection`, `higher-order-kernels`,
`iterative-linear-solvers` or `streaming-batches` (the coordinate-descent substrate is
cohesive enough to own its own capability rather than extend the reduction or ridge
families).

## Impact

- **New code:** `crates/sciencekit_linear_model/src/coordinate_descent/` (`config`,
  `solver`, `screening`, `accumulator`, `state` + companion `*_tests.rs`), `src/lasso/`
  (`builder`, `core_implementation`, `fitting_logic`, `model/`, `streaming/` + companion
  `*_tests.rs`) and `src/elastic_net/` (same shape).
- **Dependencies:** none new — `ndarray`, `sprs` and `rayon` are already workspace/production
  dependencies of `sciencekit_linear_model` (rayon is promoted to a production dependency of
  the crate for the per-alpha parallelism).
- **Public API:** new exports `SKLasso`, `SKLassoBuilder`, `SKLassoModel`, `SKElasticNet`,
  `SKElasticNetBuilder`, `SKElasticNetModel`, `SKSelection` — all additive, no existing item
  changes.
- **Downstream:** `SKLogisticRegression` (W2.4), the linear SVM and `RidgeCV` (W6) reuse the
  coordinate-descent substrate and the retained-design accumulator; `docs/src/algorithms.md`
  gains the `SKLasso`/`SKElasticNet` entries and a new Developer Book chapter documents them.
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large data; dense,
  incremental and streaming fits agree within float tolerance; sparse and dense agree within
  a documented relative tolerance; the solution satisfies the KKT conditions; safe under
  concurrency; the model carries its backend kind and is the export boundary. Model export
  (Safetensors/JSON) and the metrics suite are deferred to their Phase 7 / W6.3 changes, as
  the scaler, W2.1 and W2.2 changes already do.
