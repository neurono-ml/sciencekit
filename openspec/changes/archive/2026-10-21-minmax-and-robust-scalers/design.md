# Design

## Context

The `standard-scaler` change established the estimator template for this crate: a
builder-configured estimator distinct from a fitted model, a carried `SKBackendKind`, the
automatic execution decision, the streaming executor, and the companion `*_tests.rs` layout.
See `proposal.md` for motivation; this design only covers how the two new scalers reuse and
extend that template. The existing `higher-order-kernels` capability provides the axis
reductions (`sk_axis_sum`, `sk_axis_sum_squares`); min/max is the missing extremum reduction.

## Goals / Non-Goals

**Goals:**
- Two more feature scalers sharing one shape with `SKStandardScaler`, so a user learns the API
  once and later algorithms consume any scaler identically.
- Incremental and out-of-core fitting for both, including the robust scaler (which
  scikit-learn cannot stream).
- Exact dense results where possible (min/max, quantiles); approximation confined to the
  online path and bounded by a documented tolerance.

**Non-Goals:**
- Sparse input support (a later sparse change).
- Exact streaming quantiles (the online sketch is approximate by design).
- Python bindings and GPU/BLAS-specific paths (associated changes).
- `n_samples_seen`-weighted variants and NaN handling beyond the dense-only boundary.

## Decisions

### Decision 1 — Reuse the `standard-scaler` template verbatim

Each scaler is an immutable estimator (`SKMinMaxScaler` / `SKRobustScaler`) plus a fitted model
(`SKMinMaxScalerModel<F>` / `SKRobustScalerModel<F>`) implementing `SKFeatureTransformer<F>`,
with a builder-only surface exposing `backend` and `execution_mode`. Learned state lives in the
model; the model is the export boundary.
*Alternative rejected:* a shared generic `SKScaler` — the two scalers have different learned
state (min/max vs quantile sketches) and different flags; a common trait would either be too
thin to help or force a lowest-common-denominator API.

### Decision 2 — `MinMaxScaler` keeps exact running extrema in every regime

Dense `fit` computes per-column min/max with the new `sk_axis_min_max` kernel; `partial_fit` and
`fit_streaming` merge per-batch extrema with `min`/`max`, which are associative and exact. All
three regimes therefore agree exactly (no tolerance), mirroring how `StandardScaler`'s regimes
agree up to float rounding.
*Alternative rejected:* sampling or sketching the extrema — unnecessary, since running min/max
is O(columns) state and exact.

### Decision 3 — `RobustScaler` dense fit is exact; the online path sketches

Dense `fit` computes the median and interquantile range exactly (per-column selection over the
sorted column), matching scikit-learn. `partial_fit` and `fit_streaming` maintain a per-column
`tdigest` sketch (`tdigest 1.0.0`, locked by the wave plan) and read the quantiles from it. The
two paths are therefore *not* bit-identical; cross-regime tests assert a documented relative
tolerance, exactly as `StandardScaler` compares its dense identity against the Chan merge.
*Alternatives rejected:* always using `tdigest` (dense results would be approximate and
surprising, breaking sklearn parity); always using exact quantiles (impossible in one streaming
pass without buffering the whole column — the capability scikit-learn lacks).

### Decision 4 — One sketch per column, compression as an internal constant

The model holds a `Vec<tdigest::TDigest>` (one per column). The compression parameter is an
internal constant (a reasonable accuracy/size trade-off), not a builder option, so the public
surface stays focused; it can be promoted later if a use case needs it. The sketch is fed per
batch, so memory is bounded by `columns × compression`, independent of row count — the
out-of-core property.

### Decision 5 — `unit_variance` via an internal probit approximation

`unit_variance = true` divides the scale by the standard-normal interquantile span of the same
`quantile_range`. Rather than add a statistics dependency for the inverse normal CDF, the crate
uses a self-contained rational approximation of the probit function (Acklam's algorithm) with a
test against known values (e.g. the 25–75 span ≈ 1.34898).
*Alternative rejected:* pulling in `statrs`/`libm` for one constant per quantile range —
dependency weight for a closed-form approximation.

### Decision 6 — `feature_range` and `clip` are transform-time concerns

`MinMaxScaler` stores the learned per-column minimum and range plus the configured target
range; `transform` applies the affine map and, when `clip` is true, clamps to the target range.
`clip` is a transform-time flag only, so changing it never requires refitting.
*Alternative rejected:* baking the clipping into the learned scale — it would couple a
presentation choice into the fitted state.

### Decision 7 — `sk_axis_min_max` mirrors the sum reductions

A single kernel returns per-column `(min, max)` pairs, sequential at `parallelism = 1` and
race-free per-chunk partials otherwise, sharing the reduction grain and generic over `F`. It is
the extremum companion of `sk_axis_sum`/`sk_axis_sum_squares`.
*Alternative rejected:* folding min/max inside the scaler — duplicated across every
extremum-based algorithm (outlier detection, trees later).

### Decision 8 — Backend carried and recorded

Both scalers expose `backend(SKBackendKind)` (default `Faer`), record the configured kind in
every span, and transfer it into the fitted model, per the `backend-selection` convention.
Neither scaler needs a concrete `SKMathBackend` (its math is reductions/selection only), but
the parameter is carried for a uniform selection surface.

## Risks / Trade-offs

- **Dense vs online divergence (RobustScaler)** → documented tolerance and explicit tests; the
  dense path stays exact, the online path is approximate by construction.
- **tdigest approximation on skewed data** → cross-regime tests use deterministic datasets with
  a known distribution; a tolerance is asserted, not bit-equality.
- **Probit approximation error** → a dedicated test pins known quantile spans; the error is far
  below the sketch's own approximation.
- **f32 precision** → same posture as `StandardScaler`; the f64 path is the reference.

## Migration Plan

Additive: two new estimators in the existing crate and one new kernel. The only workspace
change is the `tdigest` dependency pin. No public API is removed or renamed.

## Open Questions

- Whether to expose the sketch's compression as a builder option later (deferred; internal
  constant for now).
- Whether a dense `sk_axis_quantile` kernel should replace the in-scaler exact-quantile helper
  if more estimators need exact dense quantiles (deferred to the outlier-detection wave).
