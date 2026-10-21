## Context

`SKRidge` is the second Wave 2 algorithm — the first whose linear system is *regularized*.
`SKLinearRegression` (W2.1) established the supervised estimator/model split, the design
matrix convention, the per-operation backend dispatch and the Gram-based streaming path. Ridge
inherits all of that and adds the parts W2.1 deliberately left out: a penalty parameter that
is a **configuration scalar** (so the estimator becomes generic over `F`), a **portfolio of
solvers** with different numerical properties, **sparse input** support, and a model that
carries **one solution per `alpha`**.

The wave plan (`wave-plan-foundation`, task 4) scopes W2.2 as *"`SKRidge` (Cholesky dense,
sparse_cg, SVD fallback, zero-copy sparse `LinearOperator`)"*. This design fixes the
numerical conventions, the solver resolution policy, the multi-alpha model shape and the
new `sciencekit_math` substrate.

## Goals / Non-Goals

- **Goals:** one estimator definition reachable through every solver and every regime
  (dense, sparse, incremental, out-of-core) that agree numerically; a builder as the only
  construction surface; an **unpenalized intercept** in every solver; sparse input that never
  densifies; a model shape that `RidgeCV` can consume later; the substrate
  (`SKLinearOperator` + `sk_conjugate_gradient`) the following sparse linear models reuse.
- **Non-Goals:** `RidgeCV` (W6); `sample_weight`, `positive=True`, `lsqr`, `sag`/`saga`,
  `lbfgs`; preconditioned CG (Jacobi) — plain CG first, per the "iterative evolution, no
  skipped steps" rule; the metrics suite and model export (W6.3 / Phase 7); Python bindings
  (associated change after CPU validation); sparse design input beyond what `SKDataView`
  already carries.

## Decisions

### Decision 1 — Estimator and model are distinct types; the estimator is generic over `F`

`SKRidge<F>` is immutable configuration (builder output) and **does** carry scalars — the
`alpha` sequence, `tolerance` — so unlike the scalar-free `SKLinearRegression` it is generic
over `F`; `SKRidgeModel<F>` is the learned state and the export boundary. This matches the
`SKRobustScaler<F>` / `SKSimpleImputer<F>` shape and PRD §3.4 ("the estimator is generic when
its configuration carries scalar values"). *Alternative rejected:* storing `alpha` as `f64`
to keep the estimator non-generic — it splits the configuration's scalar type from the
model's and is inconsistent with the preprocessing estimators.

### Decision 2 — The intercept is unpenalized, spelled two equivalent ways

Shrinking the intercept makes the fit depend on the target's origin — statistically wrong.
The Gram and SVD solvers **center** the design and targets and recover `β₀ = ȳ − x̄ᵀβ`; the CG
solver solves the **augmented** system `(DᵀD + αP)β = Dᵀy` with `D = [1 ‖ X]` and
`P = diag(0, 1, …, 1)`. Centering and the unpenalized augmentation are algebraically the
**same estimator**, so all solvers target one definition and are cross-tested for agreement.
*Alternative rejected:* W2.1's plain `[1 ‖ X]` augmentation — it would penalize the intercept
column. *Alternative rejected:* a single spelling for all solvers — centering densifies a CSR
matrix (used only by the Gram solvers, which form the dense `p × p` Gram anyway), while the
augmented form is the sparse-friendly one.

### Decision 3 — Solver portfolio and deterministic `Automatic`

`SKRidgeSolver ∈ {Automatic, Cholesky, Svd, SparseCg}`. `Automatic` resolves by
representation: dense → `Cholesky`, sparse → `SparseCg`. `Cholesky` falls back to `Svd` when
`alpha = 0` or the factor fails (a non-positive-definite centered Gram). `Svd` is dense-only
and rejects sparse input, as scikit-learn does. The solver actually used is recorded in the
span and exposed by the model, because "which solver ran" is observable behavior a consumer
may depend on. *Alternative rejected:* hiding the solver choice entirely — silently swapping
a factorization for an iterative solve changes cost and accuracy characteristics the consumer
should be able to see.

### Decision 4 — Cholesky forms the centered Gram; it serves dense and sparse

The Cholesky solver builds `Ĉ = XᵀX − n·x̄x̄ᵀ` and `ĉ = Xᵀy − n·x̄·ȳ`. For dense input these
come from the W2.1 Gram kernels; for sparse input `XᵀX` is computed from the sparse structure
(column sums for `x̄`, sparse products for `XᵀX`) without densifying the design — only the
`p × p` Gram is dense. This makes ridge's exact path available to sparse users when `p` is
small, complementing the memory-bounded CG path. *Alternative rejected:* forcing sparse users
onto CG always — unnecessary when the Gram fits in memory.

### Decision 5 — SVD applies one decomposition to every alpha

The `Svd` solver centers once, decomposes the centered design once, and applies each `alpha`
as `β = V diag(σ/(σ²+α)) Uᵀ ỹ`. The expensive factorization is amortized across the alpha
sequence — this is why multi-alpha is worth supporting here rather than deferring to a second
estimator. Rank deficiency with `alpha = 0` is precisely the case SVD exists for.

### Decision 6 — Sparse CG never forms the Gram

`SparseCg` minimizes through `SKLinearOperator` applied to a **borrowed `CsMatView`**: the
operator computes `D v` and `Dᵀ w` by walking the sparse rows and columns, so `XᵀX` is never
materialized and the design is never densified. It honors `tolerance` and
`maximum_number_of_iterations` and raises `SKError::NotConverged { iterations }` when the
budget is exhausted — never a silent wrong answer. *Alternative rejected:* densify-then-solve,
which defeats the point of accepting sparse input. *Alternative rejected:* preconditioned CG —
plain CG first (Decision 6 of the wave plan: iterative evolution, no skipped steps); Jacobi
preconditioning is a recorded follow-up.

### Decision 7 — One centered Gram accumulator drives the extra regimes

`partial_fit` and the streaming branch carry one state — `XᵀX`, `Xᵀy`, `Σx`, `Σy` and the row
count — and solve the accumulated centered normal equations once at the end, so incremental,
streaming and dense fits reduce to the same cross products and agree by construction. Ridge
introduces its own `SKCenteredGramState` beside W2.1's augmented-Gram `SKGramState`: the two
are genuinely different formulations — OLS augments with the intercept column, ridge centers —
which is exactly Decision 2, so sharing one type would force one estimator into the other's
spelling. The OLS accumulator is left unchanged. A `SparseCg` intent met in a streaming fit
falls back to this Gram solve and records the solver actually used, because a single-pass
source cannot drive a per-batch operator. *Alternative rejected:* per-batch CG — the operator
would have to be rebuilt cumulatively, and the result would depend on batch boundaries.

### Decision 8 — Backend is carried, not dispatched, by the estimator

Per `backend-selection`, the estimator exposes `backend(SKBackendKind)`, records the kind in
every span and transfers it into the model. The factorizations dispatch to a concrete
`SKMathBackend` per operation through `SKBackendSelection`, never in `build()`; the CG path
uses no backend (it is a host-side iteration over `sciencekit_math` kernels), which the design
records explicitly so the "backend recorded" requirement stays honest.

### Decision 9 — The iterative substrate is its own capability

The `SKLinearOperator` trait, `sk_conjugate_gradient` and the sparse normal-equation operator
are general SPD substrate, not ridge internals: `SKTruncatedSVD`, PCA's covariance solves and
the following regularized linear models consume them. They live in `sciencekit_math` under a
new `iterative-linear-solvers` capability (their own spec) rather than extending the
`higher-order-kernels` reduction family, which is about elementwise/axis/cross-product
kernels, not iterative solves.

### Decision 10 — `predict` is single-alpha; the multi-alpha path is explicit

`SKRegressorPredictor<F>` returns `Array1<F>` — one response per row — which is only defined
for one coefficient vector. With several alphas configured, plain `predict` returns a
structured error and the consumer calls `predict_with_alpha_index` (or reads
`coefficients()`/`intercepts()` directly). This mirrors scikit-learn, which raises when
`predict` is called on a multi-alpha `Ridge`.

### Decision 11 — Export and metrics are deferred, as W2.1 and the scalers do

No `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model is
already the distinct, backend- and solver-carrying export boundary this change establishes;
export and metric generation are tracked for those changes.

## Risks / Trade-offs

- **CG non-convergence on ill-conditioned sparse systems:** surfaced as
  `SKError::NotConverged`, never a silent wrong answer; `Automatic` only routes to CG for
  sparse input. Mitigation: expose `tolerance`/`maximum_number_of_iterations`, and record
  Jacobi preconditioning as a follow-up.
- **f32 CG and Gram stability:** the squared-condition caveat W2.1 documented for the Gram path
  applies to the CG normal-equation path too; documented, with `f64` accumulation as the
  escape hatch. Cross-solver tests use a relative tolerance.
- **Two spellings of the intercept:** documented as algebraically equivalent and pinned by a
  cross-solver agreement test, so they cannot drift into two estimators.
- **Estimator genericity:** `SKRidge<F>` is generic, unlike `SKLinearRegression`; consumers
  must annotate `F` or rely on inference from the data — consistent with the preprocessing
  estimators.

## Migration Plan

Additive: a new `ridge` module in an existing crate, a new `solvers` module in
`sciencekit_math`, one `sprs` dependency promotion (dev → production in
`sciencekit_linear_model`) and no workspace-member change. No existing item changes; no span
schema change. Ridge adds its own `SKCenteredGramState` (internal, no public change); the
W2.1 `SKGramState` is untouched.

## Discovered during implementation

_(filled in as the apply surfaces anything unexpected, as W2.1 did.)_
