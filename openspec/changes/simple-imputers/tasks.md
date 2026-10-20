# Tasks

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with a
pure-dispatcher `mod.rs`. One commit per task. ADR issue: #87.

## 1. Crate scaffold and workspace member

- [x] 1.1 Add `crates/sciencekit_impute` (root `Cargo.toml`, `src/lib.rs` with the
  `simple_imputer` and `missing_indicator` module declarations) and register it in the
  workspace `members`, with `sprs`/`tdigest` taken from the workspace pins. Verify:
  `cargo build -p sciencekit_impute` succeeds and `cargo test --workspace` stays green.

## 2. Marker semantics and the owned output type

- [x] 2.1 Red: companion tests for the marker rules (dense `NaN` marker vs explicit marker,
  CSR explicit entries, the `0.0` marker marking implicit zeros, any other marker leaving
  implicit zeros observed) and for `SKImputedOutput<F>` (`Dense`/`Sparse` variants,
  shape accessors, `#[non_exhaustive]`). Verify: tests fail.
- [x] 2.2 Implement the marker-mask helpers over `SKDataView` and the `SKImputedOutput<F>`
  enum exported from the crate root. Verify: the marker tests pass for `f32` and `f64`.

## 3. SKSimpleImputer — builder and estimator shell

- [x] 3.1 Red: companion tests for `SKSimpleImputerBuilder` (defaults: `Mean`, `NaN` marker,
  `keep_empty_features(false)`, `add_indicator(false)`, `Faer`, `Automatic`; every setter
  chainable; private constructor; invalid configuration rejected). Verify: tests fail.
- [x] 3.2 Implement `SKImputationStrategy` and `SKSimpleImputerBuilder` (plus the `SKBuilder`
  impl) and the `SKSimpleImputer` estimator shell carrying the configuration. Verify: the
  builder tests pass.

## 4. Dense fit — per-column statistics

- [x] 4.1 Red: companion tests for dense `fit` (mean over observed values, exact median of
  even and odd counts, most-frequent with the smallest-tie rule, constant, all-missing
  column producing a `NaN` statistic, column-count bookkeeping, carried backend kind,
  `fit(&self)` leaving the estimator reusable). Verify: tests fail.
- [x] 4.2 Implement the column statistics — rayon `par_iter` over columns with compensated
  sum for the mean, a sorted order statistic for the median, a count map with deterministic
  tie-break for the mode — and `SKSimpleImputerModel<F>` with its accessors. Verify: the
  dense fit tests pass.

## 5. Transform — dense and sparse, empty-column policy

- [x] 5.1 Red: companion tests for `transform` (only missing cells change and observed cells
  stay bit-identical, column-count mismatch rejected before reading elements, dense→dense and
  CSR→CSR, sparse fill storing only the filled cells, empty column dropped by default,
  retained as `0.0` under `keep_empty_features(true)`, filled under `Constant`). Verify:
  tests fail.
- [x] 5.2 Implement `transform` on the model returning `SKImputedOutput<F>` — `azip!`/`map`
  fill for dense, a two-pass CSR rebuild for sparse — plus the empty-column policy. Verify:
  the transform tests pass and sparse ≡ dense equivalence holds exactly.

## 6. Incremental fit (partial_fit)

- [x] 6.1 Red: companion tests for `partial_fit` (split uneven batches ≡ `fit` on the
  concatenation for mean/mode/constant, median equal to the exact dense value while the
  sketch path stays within tolerance, empty batch no-op, column-count mismatch error,
  chaining `&mut self`). Verify: tests fail.
- [x] 6.2 Implement `partial_fit` on the model with its accumulating state (count/sum, value
  counts, per-column `tdigest` for the median). Verify: the incremental equivalence tests
  pass.

## 7. Out-of-core fit_streaming

- [x] 7.1 Red: companion tests for `fit_streaming` (in-memory intent ≡ dense fit exactly,
  driver path within sketch tolerance for the median and exact for mean/mode, uneven batch
  sizes, empty source error, spans recording the configured backend kind). Verify: tests
  fail.
- [x] 7.2 Implement `fit_streaming` over `SKLazySource` mirroring the scaler split (collected
  dense branch vs `sk_run_streaming_driver` with per-batch updates). Verify: the streaming
  tests pass.

## 8. SKMissingIndicator — estimator, model and mask

- [x] 8.1 Red: companion tests for `SKMissingIndicator` (defaults `MissingOnly`,
  `error_on_new(true)`, dense output; scope learned at fit; deterministic scope across fits;
  mask `1.0`/`0.0` at the missing cells; sparse input → sparse mask with the missing cells as
  stored entries; `All` scope; `error_on_new` rejecting and then tolerating new missingness;
  column-count mismatch; carried backend). Verify: tests fail.
- [x] 8.2 Implement `SKMissingFeatureScope`, `SKMissingIndicatorBuilder`, the estimator and
  `SKMissingIndicatorModel<F>` (fit scope + typed mask transform + `error_on_new`). Verify:
  the indicator tests pass.

## 9. add_indicator stacking

- [ ] 9.1 Red: companion tests for the stacking rule (output width = retained features +
  counted missing features, indicator columns ordered after the features, model exposing the
  fitted indicator, disabled by default). Verify: tests fail.
- [ ] 9.2 Implement the horizontal stack inside the imputer's transform and expose the
  fitted `SKMissingIndicator` from `SKSimpleImputerModel`. Verify: the stacking tests pass.

## 10. Acceptance — small and large data, concurrency, export

- [ ] 10.1 Acceptance tests: hand-checkable tiny matrix end to end, a large generated matrix
  streamed through `fit_streaming`, the shared estimator fitted concurrently from several
  threads (`fit(&self)`) and one model used concurrently for `transform`, dense and sparse
  outputs provably equal, and the model's statistics/scope reachable as the export boundary.
  Verify: the acceptance tests pass under `cargo test --workspace`.

## 11. Documentation

- [ ] 11.1 Add the doc-branch chapters: a new `Development Explained` chapter for the impute
  crate and both estimators (with Mermaid diagrams), register it in `SUMMARY.md`, update
  `maintaining.md` and the Imputation section of `docs/src/algorithms.md`. Verify:
  `mdbook build docs` passes with zero warnings.

## 12. Gates

- [ ] 12.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets` and
  `cargo test --workspace` all green; `mdbook build docs` zero warnings.

## 13. Independent review

- [ ] 13.1 Independent review by a separate agent validating the specs were met; fix
  findings.

## Workflow follow-up

- Push the code, openspec and docs branches and open the three PRs, each referencing ADR
  #87 with a closing keyword; update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) when
  the change lands; archive the change after the PRs merge.
