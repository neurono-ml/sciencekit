# Tasks

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with a
pure-dispatcher `mod.rs`. One commit per task.

## 1. Kernel — per-column min/max reduction

- [ ] 1.1 Red: companion tests for `sk_axis_min_max` (column extrema, f32/f64, parallel ≡
  sequential exactly). Verify: tests fail before the function exists.
- [ ] 1.2 Implement `sk_axis_min_max` (sequential + race-free per-chunk parallel partials,
  shared reduction grain) and export it from the crate root. Verify: the new kernel tests pass
  and `cargo test -p sciencekit_math` is green.

## 2. MinMaxScaler — estimator, model and dense path

- [ ] 2.1 Red: companion tests for `SKMinMaxScaler` (unit-range mapping, `feature_range`
  override, constant column, `clip` on/off, empty/sparse errors, transform column mismatch,
  backend default and override, f32/f64). Verify: tests fail.
- [ ] 2.2 `SKMinMaxScalerBuilder` (`feature_range`, `clip`, `backend`, `execution_mode`;
  private constructor) and the `SKBuilder` impl. Verify: builder tests compile and default
  configuration is asserted.
- [ ] 2.3 `SKMinMaxScaler` estimator (`SKUnsupervisedFit`) and `SKMinMaxScalerModel`
  (`SKFeatureTransformer`, learned min/range/scale/min_, accessors, backend). Verify: the
  dense fit/transform tests pass.
- [ ] 2.4 Document the scaler in the companion docs change (new `ch09` Developer Book chapter
  section) and verify `mdbook build docs` passes with zero warnings.

## 3. MinMaxScaler — incremental and out-of-core

- [ ] 3.1 Red: companion tests for `partial_fit` (prefix+rest ≡ dense, empty no-op, column
  mismatch, chaining, builder flags) and `fit_streaming` (uneven batches ≡ dense, in-memory
  intent collects, empty source error). Verify: tests fail.
- [ ] 3.2 Implement running-extrema `partial_fit` and `fit_streaming` (streaming executor +
  in-memory collect branch). Verify: incremental and streaming tests pass, exactly equal to
  dense.

## 4. RobustScaler — estimator, model and dense exact path

- [ ] 4.1 Red: companion tests for `SKRobustScaler` dense behavior (median/IQR on a
  hand-checkable column, `quantile_range` override, `with_centering`/`with_scaling` masking,
  zero-spread column, `unit_variance` against known probit spans, empty/sparse errors,
  transform column mismatch, backend default/override, f32/f64). Verify: tests fail.
- [ ] 4.2 `SKRobustScalerBuilder` (`with_centering`, `with_scaling`, `quantile_range`,
  `unit_variance`, `backend`, `execution_mode`; private constructor) and the `SKBuilder` impl.
  Verify: default configuration asserted.
- [ ] 4.3 Exact dense quantile helper + the probit approximation (Acklam) with a test pinning
  the 25–75 standard-normal span (≈ 1.34898). Verify: helper tests pass.
- [ ] 4.4 `SKRobustScaler` estimator and `SKRobustScalerModel` (learned center/scale, masking,
  accessors, backend). Verify: the dense fit/transform tests pass.
- [ ] 4.5 Document the scaler in the companion docs change and verify `mdbook build docs`
  passes with zero warnings.

## 5. RobustScaler — online quantile sketch, incremental and out-of-core

- [ ] 5.1 Add the `tdigest 1.0.0` workspace pin and the `sciencekit_preprocessing` dependency.
  Verify: `cargo build -p sciencekit_preprocessing` succeeds.
- [ ] 5.2 Red: companion tests for `partial_fit` and `fit_streaming` (approximate dense within
  the documented tolerance, empty no-op, column mismatch, sample-count accumulation, in-memory
  intent collects, empty source error). Verify: tests fail.
- [ ] 5.3 Implement the per-column sketch state and the online `partial_fit`/`fit_streaming`
  path. Verify: incremental and streaming tests pass within tolerance.

## 6. Acceptance and gates

- [ ] 6.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets` and
  `cargo test --workspace` all green; `mdbook build docs` zero warnings.
- [ ] 6.2 Small and large data validated; concurrency validated (shared estimator/model across
  threads); models carry their backend kind as the export boundary; export/metrics deferred to
  the interop/metrics changes as in `standard-scaler`.
- [ ] 6.3 Independent review by a separate agent validating the spec was met; fix findings.

## Workflow follow-up

- Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) when the change lands.
- Open the code, openspec and docs PRs referencing the ADR issue; archive the change after the
  PRs merge.
