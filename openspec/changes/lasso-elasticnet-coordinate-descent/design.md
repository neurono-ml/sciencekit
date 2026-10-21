## Context

`SKLasso` and `SKElasticNet` are the third pair of Wave 2 algorithms. `SKLinearRegression`
(W2.1) established the supervised estimator/model split, the design-matrix convention, the
per-operation backend dispatch and the Gram-based streaming path. `SKRidge` (W2.2) added a
regularized objective, an unpenalized centered intercept, sparse input and a multi-`alpha`
model. Lasso/ElasticNet inherit all of that and add the parts W2.2 deliberately left out: a
penalty that is **non-differentiable at the origin**, a solver that is **iterative by
nature** (no factorization), a **sparse solution** whose exactly-zero coordinates make
screening worth doing, and an objective that is **not a function of the Gram matrix** — so
the incremental and streaming regimes cannot reuse W2.2's cross-product accumulator.

The wave plan (`wave-plan-foundation`, task 4) scopes W2.3 as *"`SKLasso` + `SKElasticNet`
(coordinate descent + gap-safe screening, rayon across multi-target)"*. This design fixes the
numeric conventions, the screening guarantee, the representation strategy and the
incremental/streaming departure from ridge.

## Goals / Non-Goals

- **Goals:** one shared coordinate-descent substrate serving both estimators; exact zeros
  from the L1 penalty; zero-copy dense **and** sparse input that never densifies a sparse
  design; an unpenalized intercept; a screened solve that is *provably identical* to the
  unscreened solve; incremental and out-of-core fits that agree with the dense fit; a model
  that `RidgeCV`-style and L1-logistic paths can consume later.
- **Non-Goals:** `LassoCV`/`ElasticNetCV` (W6); `sample_weight`; `warm_start` across
  estimators; multi-output target **matrices** (independent response columns) — see
  Decision 11; the metrics suite and model export (W6.3 / Phase 7); Python bindings
  (associated change after CPU validation); GPU backends.

## Decisions

### Decision 1 — One shared substrate, two estimators, two distinct models

The coordinate-descent solver, the screening, the selection and the retained-design
accumulator live in one `coordinate_descent` module; `SKLasso` and `SKElasticNet` are thin
builder-configured estimators over it, and their fitted models (`SKLassoModel`,
`SKElasticNetModel`) are **distinct public types** wrapping a shared internal core. Lasso is
the `l1_ratio = 1` special case of the elastic-net objective, so duplicating the solver would
let the two drift; the distinct types keep each estimator's public contract (ElasticNet's
`l1_ratio` accessor, the capability split) honest. *Alternative rejected:* one public
estimator with `l1_ratio` — scikit-learn ships two names for two statistically distinct
models, and the wave plan names both.

### Decision 2 — Estimators are generic over `F`, models are the export boundary

Both estimators carry scalar configuration (`alpha`, `tolerance`, `l1_ratio`), so they are
generic over the model scalar `F`; the learned state lives only on the distinct fitted model,
so "not yet fitted" is unrepresentable. This mirrors `SKRidge<F>`/`SKRidgeModel<F>` and PRD
§3.4.

### Decision 3 — The unpenalized intercept is the residual-mean coordinate

Shrinking the intercept makes the fit depend on the target's origin — statistically wrong.
Coordinate descent treats the intercept as one more coordinate: because it is unpenalized,
its exact update is `β₀ ← mean(r)`, applied at the end of each sweep with the residual
recentered. Keeping the design uncentered and the residual mean-zero is **algebraically
equivalent** to centering the design and recovering `β₀ = ȳ − x̄ᵀw`, but it avoids
densifying a CSR matrix (centering `x_j − x̄_j` is dense even for a sparse column).
`include_intercept(false)` drops the coordinate and solves through the origin. *Alternative
rejected:* center-then-solve (W2.2's Gram path) — that needs the design densified or the Gram
formed, both of which defeat sparse Lasso.

### Decision 4 — Soft-threshold coordinate updates, convergence on the maximum change

The update for coordinate `j` minimizes the objective in `w_j` with the others fixed:
`w_j = S(x_jᵀr/n + a_j·w_j, α·ρ) / (a_j + α(1−ρ))`, with `a_j = ‖x_j‖²/n` and `S` the
soft-threshold. Sweeps iterate the coordinates (cyclic, or a deterministic pseudo-random
order under `selection(Random)`); convergence is `max_j |Δw_j| ≤ tolerance` over a sweep,
bounded by `maximum_number_of_iterations` and surfaced as `SKError::NotConverged { iterations }`
— never a stale vector passed off as a solution. The `positive` flag clamps each update at
zero (non-negative least squares with L1/L2). *Alternative rejected:* proximal gradient /
FISTA — plain coordinate descent first, per the "iterative evolution, no skipped steps" rule.

### Decision 5 — Gap-safe screening, correct by construction

Lasso's solution is sparse, so many coordinates are exactly zero at the optimum. The solver
uses a **strong initial active set** (coordinates whose gradient at the origin exceeds the
L1 threshold), a **gap-safe safe-screening** test for the pure-L1 objective — a coordinate is
discarded only when `|x_jᵀz| + ‖x_j‖·R < α`, with `z` a dual-feasible point and
`R = sqrt(2·gap/n)` the primal–dual gap radius — and, crucially, a **KKT reactivation pass**:
after the active-set solve converges, every coordinate's full-objective gradient is checked
and any violator is added back and the solve resumes. The reactivation guarantees the final
coefficients solve the *unscreened* problem, so screening can only change how much work is
done, never the answer. The model exposes `screened_count()`. *Alternative rejected:*
screening without reactivation — a heuristic that can silently return a suboptimal answer.
*Alternative rejected:* full GAP-SAFE for the elastic net too — recorded as a follow-up;
elastic net still uses the strong set plus the same correctness guarantee.

### Decision 6 — Dense and sparse share one coordinate loop

The iteration is written once against a small column-access abstraction (`column_squared_norm`,
`signed_column_dot`, `update_residual`); the dense implementation reads contiguous columns and
the sparse one walks the stored entries of a borrowed `CsMatView`. A sparse design is never
densified: the closest thing to a dense object is the `O(n)` residual vector. *Alternative
rejected:* separate dense and sparse solvers — they would double the iteration, the screening
and the convergence logic and drift.

### Decision 7 — The retained-design accumulator backs partial_fit and fit_streaming

The L1 objective is not a function of `XᵀX` and `Xᵀy` alone, so W2.2's cross-product
accumulator cannot be reused: the incremental and streaming regimes accumulate the **design**
(dense rows appended, or CSR batches stacked) plus the targets and re-solve at read time. The
trade-off — memory grows with the data, unlike ridge's fixed `O(p²)` state — is inherent to
Lasso and is documented. The accumulator is shared by both estimators. *Alternative rejected:*
per-batch online CD — it would make the result depend on batch order and size, breaking the
"incremental ≡ dense" guarantee.

### Decision 8 — Backend is carried and recorded, not dispatched by the solver

Per `backend-selection`, both estimators expose `backend(SKBackendKind)` (default `Faer`),
record the kind in every span and transfer it into the model; the kind is never hardcoded.
The coordinate-descent iteration is host-side over `ndarray`/`sprs` arithmetic and dispatches
no concrete `SKMathBackend`, which the design records explicitly so the "backend recorded"
requirement stays honest (as W2.2 recorded for its CG path).

### Decision 9 — The substrate owns its capability

The coordinate-descent objective, validation, selection, screening and accumulator are
general L1/L2-regularized-regression substrate, not lasso internals: logistic regression,
the linear SVM and `RidgeCV` reuse them. They live under a new `coordinate-descent-solver`
capability (their own spec) rather than extending `higher-order-kernels` (elementwise/cross
kernels) or `iterative-linear-solvers` (`sciencekit_math` CG).

### Decision 10 — `predict` is single-alpha; the multi-alpha path is explicit

`SKRegressorPredictor<F>` returns one response per row, defined only for one coefficient
vector. With several alphas configured, plain `predict` returns a structured error and the
consumer calls `predict_with_alpha_index` (or reads `coefficients()`/`intercepts()`), exactly
as `SKRidgeModel` does.

### Decision 11 — Parallelism is across independent per-alpha solves

The wave plan's "rayon across multi-target" is realized as rayon parallelism across the
**independent per-`alpha` solves** (each alpha is a self-contained L1-regularized problem),
with results independent of the thread count. Full multi-output target matrices (independent
response columns) are a separate concern (scikit-learn's `MultiTaskLasso`/`MultiTaskElasticNet`
are distinct estimators) and are deferred rather than folded into this change. Single-target
`fit` follows `SKSupervisedFit` like W2.1/W2.2.

### Decision 12 — Export and metrics are deferred, as W2.1, W2.2 and the scalers do

No `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model
is already the distinct, backend-carrying export boundary this change establishes; export and
metric generation are tracked for those changes.

## Risks / Trade-offs

- **Coordinated-descent non-convergence on ill-conditioned designs:** surfaced as
  `SKError::NotConverged`, never a silent wrong answer; `tolerance` and
  `maximum_number_of_iterations` are exposed. Mitigation: document tighter tolerances and the
  `f64` escape hatch.
- **Screening correctness:** rests on the KKT reactivation pass; the gap-safe radius only
  reduces work. A dedicated test forces a coordinate to be wrongly screened and asserts the
  reactivation recovers the unscreened solution.
- **f32 coordinate descent:** accumulation in `f32` loses precision faster than the dense
  factorizations; documented, cross-tested with a looser relative tolerance, `f64` recommended.
- **Retained design memory:** incremental/streaming fits hold the design, unlike ridge;
  documented as the L1 price, with `partial_fit`/`fit_streaming` still order-independent.
- **Estimator genericity:** `SKLasso<F>`/`SKElasticNet<F>` are generic like `SKRidge<F>`;
  consumers annotate `F` or rely on inference.

## Migration Plan

Additive: new `coordinate_descent`, `lasso` and `elastic_net` modules in an existing crate
and one `rayon` dependency promotion (workspace → production in `sciencekit_linear_model`);
no workspace-member change. No existing item changes; no span schema change.

## Discovered during implementation

_(filled in as the apply surfaces anything unexpected, as W2.1 and W2.2 did.)_
