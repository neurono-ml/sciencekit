## Why

Wave 1 (PRD §12 Phase 1) opens with `SKStandardScaler`: the first algorithm crate,
`sciencekit_preprocessing`, and the reference implementation every later estimator copies.
It exercises the whole Phase 0 substrate end to end — the builder pattern, the explicit
backend parameter (`backend-selection`), the automatic execution decision, the streaming
executor, and the observability spans — and it proves the reduction kernels carry real
algorithms, not just synthetic tests. It also introduces the missing `Σx²` reduction
(`sk_axis_sum_squares`) the variance computation needs, and the incremental/out-of-core
paths (PRD §4.4, the library's differentiator) on the simplest possible estimator.

Doing the scaler first is deliberate: its math is trivial (per-column reductions), so any
incoherence in the estimator/model split, the backend plumbing or the streaming contract
surfaces here, before it is copied into a dozen algorithms.

## What Changes

- **New crate `crates/sciencekit_preprocessing`** with `SKStandardScaler` split into an
  immutable estimator (configuration) and a fitted `SKStandardScalerModel` (learned state
  and export boundary).
- **Builder surface:** `SKStandardScaler::build()` exposes `with_mean(bool)`,
  `with_std(bool)`, `backend(SKBackendKind)` (default `Faer`) and
  `execution_mode(SKExecutionMode)` (default `Automatic`); the direct constructor stays
  private.
- **Dense fit + transform:** `fit` learns per-column mean and population scale from
  `Σx`/`Σx²` under a `Fit` observability span; `transform` centers/scales with
  `azip!` under a `Transform` span. Empty inputs fail before any element is processed;
  sparse inputs are rejected at the dense-only boundary.
- **Incremental `partial_fit`:** the fitted model merges one more batch into running
  moments with the exact Chan–Golub–LeVeque merge, converging to the dense result
  regardless of batch order or size (a capability scikit-learn's `StandardScaler` has but
  only via `partial_fit`; here it is first-class).
- **Out-of-core `fit_streaming`:** a `SKLazySource` is consumed through the Phase 0
  streaming driver (I/O ∥ CPU), accumulating per-batch moments; an in-memory execution
  intent collects the batches and reuses the dense path.
- **Kernel `sk_axis_sum_squares`** in `sciencekit_math`: the `Σx²` companion of
  `sk_axis_sum`, same parallelism contract and `SK_AXIS_SUM_GRAIN`, sequential and
  race-free parallel paths.
- **Published capability `backend-selection`** and its builder rule in `AGENTS.md`: the
  backend is an explicit, closed-enum builder parameter carried from builder to estimator
  to model, recorded in every operation's span, never hardcoded.
- **Housekeeping:** `SKOperationAttributes` shape fields renamed to
  `rows_count`/`columns_count` (they are counts, not the rows), keeping the span field
  names `rows`/`columns` unchanged.

## Capabilities

### New Capabilities

- `standard-scaler`: the `SKStandardScaler` estimator/model contract — builder surface and
  defaults, dense `fit`/`transform` semantics (per-column mean and population standard
  deviation, zero-variance columns, `with_mean`/`with_std` masking), error taxonomy, the
  incremental `partial_fit` merge, and the out-of-core `fit_streaming` equivalence.

### Modified Capabilities

- `higher-order-kernels`: adds the `Σx²` axis reduction (`sk_axis_sum_squares`), the
  variance-style companion of `sk_axis_sum`, with the same parallelism and grain contract.

## Impact

- **Code:** new crate `crates/sciencekit_preprocessing`
  (`standard_scaler/{builder,core_implementation,model,streaming}` + companion
  `*_tests.rs`); `crates/sciencekit_math/src/kernels/reductions.rs` gains
  `sk_axis_sum_squares`; `sciencekit_common` observability shape fields renamed.
- **Workspace:** `crates/sciencekit_preprocessing` added to `[workspace] members`; the
  workspace `ndarray` pin gains the `rayon` feature.
- **Dependencies:** none new beyond the existing `ndarray`/`sprs`; the streaming path
  reuses the Phase 0 executor.
- **Downstream:** `minmax-and-robust-scalers`, `one-hot-and-polynomial-encoders` and every
  later estimator follow this crate/estimator/model layout and the `backend-selection`
  surface; `partial_fit` + `fit_streaming` become the template for out-of-core algorithms.
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large
  (100 000 × 8) data, streaming ≡ dense within float tolerance, incremental ≡ dense,
  safe under concurrency, model carries its backend kind and is exportable.
