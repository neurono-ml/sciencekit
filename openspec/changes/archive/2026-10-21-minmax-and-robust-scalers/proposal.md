# Proposal

## Why

Wave 1 (PRD §12 Phase 1) continues past `SKStandardScaler` with the other two feature
scalers. `SKMinMaxScaler` maps every feature to a bounded range — required by algorithms that
assume bounded inputs (distance metrics, neural-style models, image features) — while
`SKRobustScaler` centers and scales on quantile statistics (median and interquartile range)
so that outliers stop dominating the scale, which plain mean/standard-deviation scaling
cannot do. Both are everyday scikit-learn transformers, and the wave plan (`wave-plan-foundation`,
W1) pairs them in one change.

The distinguishing opportunity is `SKRobustScaler`'s **online quantile sketch** (`tdigest 1.0.0`,
locked by the wave plan's spike): scikit-learn's `RobustScaler` cannot be fitted incrementally
because `np.nanpercentile` needs the whole column in memory; with an online sketch, this
library can offer `partial_fit` and out-of-core streaming for the robust scaler as well — the
out-of-core differentiator applied to a transformer scikit-learn cannot stream.

## What Changes

- **New `SKMinMaxScaler`** in `sciencekit_preprocessing`, following the `standard-scaler`
  template (distinct estimator/model, builder-only construction, carried backend, companion
  tests). Builder: `feature_range((min, max))` defaulting to `(0, 1)`, `clip(bool)`
  defaulting to `false`, `backend(SKBackendKind)`, `execution_mode(...)`. Dense `fit`/`transform`,
  incremental `partial_fit` via running per-column minima/maxima, and out-of-core
  `fit_streaming`.
- **New `SKRobustScaler`** in `sciencekit_preprocessing`. Builder: `with_centering(bool)`
  defaulting to `true`, `with_scaling(bool)` defaulting to `true`, `quantile_range((low, high))`
  defaulting to `(25.0, 75.0)`, `unit_variance(bool)` defaulting to `false`,
  `backend(SKBackendKind)`, `execution_mode(...)`. Dense `fit`/`transform` use exact per-column
  quantiles (scikit-learn parity); `partial_fit` and `fit_streaming` use the online `tdigest`
  sketch.
- **New kernel `sk_axis_min_max`** in `sciencekit_math`: per-column minimum and maximum in one
  pass, with the same parallelism contract as `sk_axis_sum` (sequential at `parallelism = 1`,
  race-free per-chunk partials otherwise), reused by the dense `MinMaxScaler` path.
- **New workspace dependency `tdigest 1.0.0`** (Apache-2.0), confined to `sciencekit_preprocessing`
  and gated to the robust scaler.
- Both scalers carry the configured `SKBackendKind` and record it in every operation span,
  per the `backend-selection` convention established by the `standard-scaler` change.

## Capabilities

### New Capabilities

- `minmax-scaler`: `SKMinMaxScaler` behavior — builder surface and defaults, dense
  fit/transform to a configurable feature range, `clip`, constant-feature handling, the
  incremental `partial_fit` (running min/max) and out-of-core `fit_streaming` equivalence, and
  the error taxonomy.
- `robust-scaler`: `SKRobustScaler` behavior — builder surface and defaults, dense
  fit/transform on median and interquartile range with `quantile_range`, `with_centering` /
  `with_scaling` / `unit_variance` masking, zero-spread handling, the online-quantile
  `partial_fit` and out-of-core `fit_streaming`, and the error taxonomy.

### Modified Capabilities

- `higher-order-kernels`: adds the per-column min/max axis reduction (`sk_axis_min_max`), the
  extremum companion of the sum/sum-of-squares reductions, with the same parallelism and grain
  contract.

## Impact

- **Code:** `crates/sciencekit_preprocessing` gains `minmax_scaler/` and `robust_scaler/`
  folder modules (builder, core implementation, model, streaming, companion `*_tests.rs`);
  `crates/sciencekit_math/src/kernels/reductions.rs` gains `sk_axis_min_max`.
- **Dependencies:** `tdigest 1.0.0` added to the workspace pins and to
  `sciencekit_preprocessing`.
- **Downstream:** distance-based and regularized models (W2/W3) consume both scalers; the
  online-quantile pattern is reused by later quantile-based estimators (outlier detection,
  W5).
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large data, streaming ≡
  dense within a documented tolerance, incremental ≡ dense, safe under concurrency, and the
  fitted model carries its backend kind as the export boundary.
