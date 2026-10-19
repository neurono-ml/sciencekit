## Context

`SKStandardScaler` is the first Wave 1 algorithm and the first consumer of the Phase 0
substrate (builders, `backend-selection`, execution resolution, the streaming executor,
observability). Its math is deliberately simple — per-column reductions — so the design
focus is not the arithmetic but the *shape* of an estimator: how configuration, learned
state, execution and export are separated, and how the dense, incremental and out-of-core
regimes are guaranteed to agree.

## Goals / Non-Goals

- **Goals:** one model shape reachable from three regimes (dense, incremental,
  out-of-core) that agree numerically; a builder that is the only construction surface; the
  backend and execution intent carried explicitly and recorded; the conventions later
  estimators will copy.
- **Non-Goals:** sparse input support (a later sparse change); GPU/BLAS-specific paths
  (accelerator changes); Python bindings (associated change after CPU validation);
  `n_samples_seen` axis semantics beyond per-column statistics.

## Decisions

### Decision 1 — Estimator and model are distinct types

`SKStandardScaler` is immutable configuration (builder output); `SKStandardScalerModel<F>`
is the learned state and the export boundary. `fit` produces a model; `transform`,
`partial_fit` and `sample_count`/`mean`/`scale`/`backend` live on the model. This matches
scikit-learn's `fit` → fitted estimator split and gives one obvious serialization target.
*Alternative rejected:* a single mutable estimator holding optional state (sklearn's
internal `_fit` flag) — it makes "not yet fitted" a runtime possibility and muddies the
export boundary.

### Decision 2 — Running moments are the single source of truth

The model stores `sample_count`, per-column `moments_mean` and `sum_squared_deviations`;
the public `mean`/`scale` arrays are *derived* from them by `derive_statistics` on every
constructor and every `partial_fit`. This is what makes incremental and batch fits
converge: both reduce to the same three quantities, and the masked `mean`/`scale` are a
pure function of them plus the `with_mean`/`with_std` flags.
*Alternative rejected:* storing `mean`/`scale` directly and accumulating them — the
incremental merge would then have to un-apply the masking to recover moments, and
zero-variance handling would drift between regimes.

### Decision 3 — One exact merge (Chan–Golub–LeVeque) for every regime

Dense fit computes `Σx` and `Σx²` with the reduction kernels and derives
`Σ(x−μ)² = Σx² − 2μΣx + nμ²`; streaming and `partial_fit` compute the same per-batch
moments and merge them with the exact Chan–Golub–LeVeque update
(`mean += δ·n_b/n`, `M2 += M2_b + δ²·n·n_b/n`). Because the merge is associative and exact
up to float rounding, batch order and sizes do not change the result — the property the
streaming and incremental tests assert. The dense path is expressed through the same
`Σx`/`Σx²` identity so the two regimes share one arithmetic, not two.
*Alternative rejected:* a naive two-pass dense fit (compute mean, then a second pass for
variance) — it would diverge from the streaming formula and double the passes.

### Decision 4 — Out-of-core is the same model, reached through the executor

`fit_streaming` resolves the plan against an *unbounded* context (`dataset_size_bytes =
u64::MAX`), so `Automatic` takes the streaming branch; an explicit
`InProcessSynchronous`/`InProcessAsynchronous` intent instead collects the batches and
reuses the dense path. The streaming branch drives the Phase 0 streaming executor with a
per-batch update closure that merges moments into `StreamingMoments`. Both branches build
the *same* `SKStandardScalerModel`, so out-of-core is a transport choice, not a different
model. A single-pass source reports zero dimensions on the outer span (totals unknown
upfront); each batch reports its real dimensions on a `PartialFit` span.

### Decision 5 — Zero-variance and masking are decided once, in `finalize_column`

A column whose variance is zero (or non-positive under rounding) gets scale `1`, and
`with_std = false` also yields scale `1`; `with_mean = false` yields mean `0`. This single
helper is the only place masking is applied, so dense, streaming and incremental outputs
cannot disagree on edge columns. The large dataset deliberately pins one constant column
to exercise this at scale.

### Decision 6 — Backend is carried, not dispatched, by the scaler

Per `backend-selection`, the scaler exposes `backend(SKBackendKind)` and records the
configured kind in every span (estimator operations) and carries it into the model
(model operations). The scaler's math is reductions only, so it never needs a concrete
`SKMathBackend`; carrying the parameter anyway keeps the selection surface uniform, so a
future GPU/BLAS scaler path is a dispatch change with no signature change.

### Decision 7 — `Σx²` is a first-class kernel

Variance needs `Σx²`; rather than square in the algorithm (an extra allocation per fit),
`sk_axis_sum_squares` mirrors `sk_axis_sum` exactly — axis 0/1, `parallelism = 1`
sequential, `> 1` race-free per-chunk partials, `SK_AXIS_SUM_GRAIN`. Kernels stay the
reusable substrate; algorithms compose them.

### Decision 8 — Explicit `rows_count`/`columns_count`

The observability shape attributes are counts, so they are named `rows_count` /
`columns_count` (the "explicit names" convention). The *span field* names remain
`rows`/`columns` — the trace schema is unchanged; only the Rust identifiers are clearer.

## Risks / Trade-offs

- **f32 precision:** `Σx²` on large f32 data loses precision faster than `f64`. Accepted
  for now; the incremental path and a Welford-style streaming accumulator are the escape
  hatch if f32 variance proves inadequate in practice.
- **`u64::MAX` sentinel for streaming resolution:** pragmatic but a little opaque; a
  dedicated `SKStreamDecision`-style hint could replace it later. It is isolated in one
  function (`resolve_streaming_plan`).
- **Two `Σx`/`Σx²`-based formulas** (dense identity vs Chan merge) are algebraically equal
  but not bit-identical; cross-regime tests therefore use a relative tolerance
  (`assert_close_with_tolerance`, 1e-9) while same-regime tests use 1e-12.

## Migration Plan

Additive: a new crate and a new kernel. The only touch on existing code is the
`SKOperationAttributes` field rename (`rows`/`columns` → `rows_count`/`columns_count`),
which is internal to the workspace and pre-release; no span schema change.

## Open Questions

- Whether `partial_fit` should live on the model or on the estimator returning a new model
  (current: on the model, mutating and returning `&mut Self` for chaining). Revisit if a
  later pipeline needs an immutable estimator-returning form.
- Sparse input support and `n_samples_seen`-weighted variants — deferred to a sparse
  preprocessing change.
