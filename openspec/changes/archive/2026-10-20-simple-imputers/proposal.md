# Proposal

## Why

Numeric features still have no missing-data story: `SKDataView` carries `ArrayView2<F>` /
`CsMatView<F>` and a float cell that happens to be `NaN` flows untouched into every
downstream estimator, where it poisons sums, means and matrix products. Wave 1 (Preprocessing
& Impute) was planned to close with the first imputer — the wave plan schedules
`simple-imputers` right after the encoders, the PRD reserves `sciencekit_impute` for exactly
this (§3.1 tree, Phase 1 item 6), and the Wave 2 linear models need a concrete preprocessor
before they land.

## What Changes

- New workspace crate `crates/sciencekit_impute` (PRD §3.1: one crate per change) holding the
  missing-data estimators, depending on `sciencekit_common`, `sciencekit_math`, `sprs` and
  `tdigest`.
- `SKSimpleImputer` (builder-only): fills missing cells per column with the
  `SKImputationStrategy` strategies `Mean` (default), `Median`, `MostFrequent` and
  `Constant`, configured through `strategy`, `fill_value`, `missing_values` (default `NaN`),
  `keep_empty_features`, `add_indicator`, `backend` (default `Faer`) and `execution_mode`
  (default `Automatic`). `fit` returns a distinct `SKSimpleImputerModel` that owns the
  per-column statistics and the dropped-column bookkeeping.
- Sparse-aware imputation via `sprs`: CSR input yields CSR output; explicit entries equal to
  the marker are missing, and a `0.0` marker additionally makes the implicit zeros missing
  (sklearn sparse idiom); otherwise implicit zeros are observed values.
- `partial_fit` and `fit_streaming` on the imputer, mirroring the scaler pattern: exact
  accumulators for `Mean`/`MostFrequent`/`Constant`, a `tdigest` sketch for `Median` under
  the streaming driver, collected dense path when the execution intent is in-memory.
- `SKMissingIndicator` (builder-only): learns which columns carried missing values at fit
  time and emits their mask at transform time, with `features` scope (`MissingOnly`
  default / `All`), `error_on_new` (default `true`), `sparse_output`, `backend` and
  `execution_mode`.
- Representation-preserving output: the imputer's `SKFeatureTransformer::Output` is an owned
  `SKImputedOutput<F>` (`Dense(Array2<F>)` | `Sparse(CsMat<F>)`), and the indicator emits
  `0.0`/`1.0` in `F` instead of sklearn's `bool`.
- Companion test modules (`*_tests.rs`) covering dense/sparse, every strategy, all-missing
  columns, incremental equivalence, concurrency and the model export boundary.

## Capabilities

### New Capabilities

- `simple-imputer`: the univariate imputation contract — marker semantics per
  representation, per-column strategy statistics, all-missing column policy, incremental and
  streaming fit equivalence, and the representation-preserving transform.
- `missing-indicator`: the missingness mask contract — feature scope learned at fit,
  `error_on_new` behavior, mask representation and its stacking onto an imputed output.

### Modified Capabilities

- (none — `data-view-boundary`, `estimator-contracts`, `backend-selection` and
  `streaming-batches` already cover the seams this change plugs into; the change is purely
  additive.)

## Impact

- **New code**: `crates/sciencekit_impute/` (builder, core, model, streaming, indicator,
  tests) and one new `[workspace] members` entry in the root `Cargo.toml`.
- **Dependencies**: `sprs` and `tdigest` reused from the workspace pins; no new pin.
- **Public API**: new exports `SKSimpleImputer`, `SKSimpleImputerBuilder`,
  `SKSimpleImputerModel`, `SKImputationStrategy`, `SKImputedOutput`, `SKMissingIndicator`,
  `SKMissingIndicatorBuilder`, `SKMissingIndicatorModel`, `SKMissingFeatureScope` — all
  additive, no existing item changes.
- **Downstream**: Wave 2 linear models gain a concrete preprocessor; `docs/src/algorithms.md`
  "Imputation · `sciencekit_impute`" stops being a placeholder; a new Developer Book chapter
  and API pages document the estimators.
- **Issue**: ADR #87 closes with the code PR.
