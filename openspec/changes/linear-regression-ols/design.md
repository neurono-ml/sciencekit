## Context

`SKLinearRegression` is the first Wave 2 algorithm and the first *supervised* estimator in
the library. It is the reference for the supervision axis of the fit contract
(`SKSupervisedFit`), the design-matrix convention every linear model shares, and the honest
streaming reformulation of a dense linear system. Its dense math is a single least-squares
solve through the backend's `lstsq`; the design focus is the shape of a supervised
estimator — how configuration, learned state, the intercept, and the dense / incremental /
out-of-core regimes are separated and kept in agreement.

## Goals / Non-Goals

- **Goals:** one model reachable from three regimes (dense, incremental, out-of-core) that
  agree numerically; a builder that is the only construction surface; the intercept solved
  jointly with the coefficients; rank and singular values surfaced rather than hidden; the
  backend and execution intent carried explicitly and recorded; the conventions later
  supervised estimators copy.
- **Non-Goals:** sparse input (a later sparse change); regularized variants — ridge, lasso,
  elastic net (their own changes); the metrics suite and model export (W6.3 / Phase 7);
  Python bindings (associated change after CPU validation); sample weights, multi-target
  regression and non-negativity constraints.

## Decisions

### Decision 1 — Estimator and model are distinct types; the estimator is scalar-free

`SKLinearRegression` is immutable configuration (builder output) with no scalar field, so
it is **not** generic over `F`; `SKLinearRegressionModel<F>` is the learned state and the
export boundary. `fit` produces a model; `predict`, `partial_fit` and the
`coefficients`/`intercept`/`rank`/`singular_values`/`backend` accessors live on the model.
This mirrors the `SKStandardScaler` split and matches scikit-learn's `fit` → fitted
estimator split. *Alternative rejected:* a single mutable estimator holding optional state
(sklearn's internal `_fit` flag) — it makes "not yet fitted" a runtime possibility and
muddies the export boundary.

### Decision 2 — Dense fit goes through the backend's `lstsq`, never the normal equations

The dense solve forms the design matrix `A` and calls `SKMathBackend::lstsq` (the
SVD/pseudoinverse path). The textbook alternative — solving `AᵀAβ = Aᵀy` with `solve` — is
rejected because `κ(AᵀA) = κ(A)²`: it squares the condition number and silently loses half
the significant digits exactly on the correlated-feature data practitioners care about. SVD
costs more but additionally returns the rank and the singular values, which the model
surfaces. *Alternative rejected:* QR — stable and cheaper, but it does not report rank for
free; kept as a future profile-driven escape hatch.

### Decision 3 — The intercept is a design-matrix column, not a separate solve

When `include_intercept` is true, a column of ones is prepended to the design matrix so the
intercept is one more coefficient solved jointly. *Alternative rejected:* centering features
and targets and solving the intercept from the means — it collapses rank prematurely on
rank-deficient designs and complicates the streaming accumulator. The design-matrix form is
the same object the streaming path accumulates.

### Decision 4 — One Gram accumulator is the single source of truth for the extra regimes

`partial_fit` and the streaming branch carry the same state — the Gram matrix `AᵀA`, the
right-hand side `Aᵀy` and the accumulated row count — and solve once at the end. This is
what makes incremental and streaming converge to the dense result: all three reduce to the
same cross products, and the coefficients are a pure function of them. *Alternative
rejected:* solving least squares per batch and averaging the coefficient vectors — a
weighted average hides how rows were distributed and per-batch ranks do not compose.

### Decision 5 — The streaming trade-off is stated, not hidden

The Gram reformulation re-introduces the squared condition number Decision 2 avoids: it is
the price of a memory-bounded (`O(p²)`) stream. The design accepts it explicitly, documents
it in the algorithm's chapter, and keeps the dense path on SVD, so a consumer who needs the
best-conditioned fit uses `fit` and a consumer who needs out-of-core uses `fit_streaming`.
The two regimes are not "the same computation"; they are two honest points on a trade-off.

### Decision 6 — Out-of-core is the same model, reached through the executor

`fit_streaming` resolves the plan against an unbounded context so `Automatic` takes the
streaming branch; an explicit in-memory intent instead collects the batches and reuses the
dense path. Both branches build the same `SKLinearRegressionModel`, so out-of-core is a
transport choice, not a different model. A single-pass source reports zero dimensions on
the outer span (totals unknown upfront); each batch reports its real dimensions on a
`PartialFit` span. This mirrors the scaler's `resolve_streaming_plan`.

The Phase 0 streaming executor carries features only (`SKDataBatch`), but a supervised
out-of-core fit needs the targets alongside them. Rather than conflate a target column into
the feature matrix or duplicate the prefetch machinery, the change adds a **supervised
streaming contract** to `sciencekit_common` — `SKSupervisedBatch` (owned features +
targets), the `SKSupervisedLazySource` fallible iteration, and
`sk_run_supervised_streaming_driver` — and generalizes the driver so both the feature-only
and the supervised sources share one prefetch engine. The feature-only path is unchanged.

### Decision 7 — Backend is carried, not dispatched, by the estimator

Per `backend-selection`, the estimator exposes `backend(SKBackendKind)` and records the
configured kind in every span, transferring it into the model. Unlike a scalar-free scaler,
the regression's dense solve *does* need a concrete `SKMathBackend`; dispatch happens per
operation at fit/predict time, never in `build()`. The estimator stays non-generic and the
model stays generic over `F`; the backend stays a runtime value.

### Decision 8 — Cross products are first-class kernels

The Gram accumulation needs `XᵀX` and `Xᵀy`; rather than transpose-and-`gemm` in the
algorithm (extra allocations and layout churn per batch), `sk_gram_matrix` and
`sk_gram_vector` join the reduction family with the same sequential / race-free parallel
contract and shared grain. They are reusable substrate: ridge and PCA's covariance
accumulator will consume the same kernels.

### Decision 9 — Rank and singular values are surfaced

`SKLeastSquaresSolution` already carries `rank` and `singular_values`; the model exposes
them. A rank below the design width is information (collinear features), not an error, and
hiding it would be a subtraction bug in plain sight. The minimum-norm solution is the
correct answer under rank deficiency and is what `lstsq` returns.

### Decision 10 — Export and metrics are deferred, as the scaler changes do

No `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model
is already the distinct, backend-carrying export boundary this change establishes; export
and metric generation are tracked for those changes, exactly as `standard-scaler` records.

## Risks / Trade-offs

- **f32 Gram accumulation:** accumulating `BᵀB` in `f32` loses precision faster than in
  `f64`; the streaming path therefore inherits the squared-condition risk twice over on
  `f32`. Accepted and documented; the escape hatch is an `f64` accumulator or a
  Welford-style reformulation if `f32` streaming proves inadequate.
- **`u64::MAX` sentinel for streaming resolution:** pragmatic but a little opaque; isolated
  in one function (`resolve_streaming_plan`), as in the scaler.
- **Cross-regime tolerance:** dense SVD and the Gram solve are algebraically equal but not
  bit-identical; cross-regime tests use a relative tolerance while same-regime tests are
  exact.

## Migration Plan

Additive: a new crate, two new kernels, and one `[workspace] members` entry. No existing
item changes; no span schema change.

## Discovered during implementation

- **The backend's least-squares kernel used the full SVD.** `SKFaerBackend::lstsq` called
  faer's `svd()`, which computes a **full** decomposition whose `U` is `m × m`. For the tall
  design matrices least squares actually sees, that allocates and computes an `m × m` matrix
  — a 20 000 × 7 design produced a 20 000 × 20 000 `U` and took ~12 s in a debug build. The
  fix is to use the **thin** SVD (`thin_svd`), which yields `U` as `m × min(m, n)`; the
  pseudoinverse path only ever needs the first `min(m, n)` singular vectors. This is a
  performance bug fix in `sciencekit_math`, pinned by a tall-design `lstsq` regression test;
  the public contract of `lstsq` (the minimum-norm solution) is unchanged, so no spec delta is
  required. A companion workspace setting (`[profile.dev.package."*"] opt-level = 2`) keeps
  dependency code optimized in dev/test builds, so the pure-Rust linear algebra does not make
  the large-data tests impractical while our own crates stay unoptimized for fast iteration.
