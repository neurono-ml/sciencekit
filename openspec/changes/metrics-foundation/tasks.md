# Tasks

## 1. Workspace and central errors

- [x] 1.1 Write failing tests in `sciencekit_common/src/errors/errors_tests.rs` for the three new
  variants — `EmptyInput` naming the operation, `NonFiniteValue { position }` and
  `UnknownLabel { label }` — then add them to the `#[non_exhaustive]` `SKError`; verify with
  `cargo test -p sciencekit_common errors`.
- [x] 1.2 Create `crates/sciencekit_metrics` with `lib.rs` and the dispatcher `mod.rs` files for
  `classification`, `regression`, `scorers` and `accumulators`, add it to
  `[workspace.members]` and workspace dependencies; verify the crate compiles empty with
  `cargo test -p sciencekit_metrics`.

## 2. Pairwise distances in `sciencekit_math`

- [x] 2.1 Write `pairwise/chebyshev_tests.rs` covering identical rows, the largest coordinate
  dominating, non-negativity, symmetry, a large input and unequal row counts, confirm the
  failure, then implement the sequential `sk_chebyshev_distance_matrix` with `azip!` and
  re-export it from `pairwise/mod.rs`; verify with `cargo test -p sciencekit_math pairwise`.
- [x] 2.2 Add the rayon row-parallel path to the Chebyshev kernel and a test asserting the
  parallel and sequential results are equal on a large input; verify with
  `cargo test -p sciencekit_math pairwise`.
- [x] 2.3 Write `pairwise/nan_euclidean_tests.rs` for the spec scenarios — co-observed-only
  comparison, the `total_features / present_features` scale, zero co-observed features
  yielding NaN, all-present rows equal to Euclidean, symmetry and large data — confirm the
  failure, then implement the sequential `sk_nan_euclidean_distance_matrix` and re-export
  it; verify with `cargo test -p sciencekit_math pairwise`.
- [x] 2.4 Add the rayon row-parallel path to the NaN-aware kernel and a test asserting parallel
  equals sequential on a large input, including rows with missing values; verify with
  `cargo test -p sciencekit_math pairwise`.

## 3. Shared evaluation validation and label space

- [x] 3.1 Implement `classification/label_space.rs`: explicit list validation (non-empty,
  unique, internally sorted) and the default sorted union of observed labels, returning
  `UnknownLabel` outside an explicit space and `InvalidHyperparameter` for duplicates; cover
  every branch in `label_space_tests.rs`; verify with `cargo test -p sciencekit_metrics`.
- [x] 3.2 Implement the shared non-finite guard over target/prediction views, returning
  `NonFiniteValue { position }`, with tests for NaN and infinity at a known position and for
  accepted finite input; verify with `cargo test -p sciencekit_metrics`.

## 4. Classification metrics

- [x] 4.1 Implement `sk_confusion_matrix` returning `SKConfusionMatrix` (row-major counts plus
  the resolved labels), with tests for known counts, zero-count resolved classes, explicit
  label ordering and unknown labels; verify with `cargo test -p sciencekit_metrics`.
- [x] 4.2 Implement `sk_accuracy_score` with tests for the known `0.75` example, perfect and
  zero accuracy, and `EmptyInput` over no observations; verify with
  `cargo test -p sciencekit_metrics`.
- [x] 4.3 Implement per-class precision, recall and F1 from the confusion matrix, with the
  harmonic-mean definition and the zero-denominator-zero policy, covered by tests for an
  absent predicted class and an absent true class; verify with
  `cargo test -p sciencekit_metrics`.
- [x] 4.4 Implement `SKAveraging` aggregation (`Macro`, `Micro`, `Weighted`,
  `Binary { positive_label }`), with tests proving macro equal class weights, micro F1 equal
  to accuracy on single-label multiclass, weighted support proportionality and the binary
  positive-label result equal to that class's per-class score; verify with
  `cargo test -p sciencekit_metrics`.
- [x] 4.5 Implement `sk_precision_score`, `sk_recall_score` and `sk_f1_score` over the
  aggregation layer, with tests matching hand-computed multiclass examples for every
  averaging mode; verify with `cargo test -p sciencekit_metrics`.

## 5. Regression metrics

- [x] 5.1 Implement the Neumaier compensated accumulator helper in `regression` and prove it
  beats naive f32 summation on a large mixed-magnitude series with a unit test; verify with
  `cargo test -p sciencekit_metrics`.
- [x] 5.2 Implement `sk_mean_squared_error`, `sk_root_mean_squared_error` and
  `sk_mean_absolute_error`, with tests for perfect predictions, the `[1,2,3]` / `[1,2,4]`
  example, length mismatch and empty input; verify with `cargo test -p sciencekit_metrics`.
- [ ] 5.3 Implement `sk_r2_score` with the zero-total-sum-of-squares policy (one for zero
  residual, zero otherwise), covered by tests for a perfect fit, a known non-perfect fit and
  constant targets; verify with `cargo test -p sciencekit_metrics`.

## 6. Scorer adapters over the shared contracts

- [ ] 6.1 Implement `SKAccuracyScorer`, `SKPrecisionScorer`, `SKRecallScorer` and
  `SKF1Scorer` over `SKLabelScorer` for any compatible classifier, with mock classifiers
  proving the pure form scores stored labels without inference and the convenient form
  equals the pure form; verify with `cargo test -p sciencekit_metrics`.
- [ ] 6.2 Implement the regression scorer adapters over `SKSupervisedScorer` for any
  compatible regressor, with the same pure/convenient equivalence tests and a nominal-target
  rejection test; verify with `cargo test -p sciencekit_metrics`.

## 7. Streaming accumulators

- [ ] 7.1 Implement `SKClassificationAccumulator` (`observe`, `merge`, label space and
  confusion finalization) with tests proving batched equals whole-input, merging two
  partitions equals one concatenated pass, an empty accumulator returns `EmptyInput`, and
  unseen labels extend the class space deterministically; verify with
  `cargo test -p sciencekit_metrics`.
- [ ] 7.2 Implement `SKRegressionAccumulator` (count, `Σy`, `Σy²`, `Σ(y−ŷ)²`, `Σ|y−ŷ|`,
  `merge`, metric finalization) with the same batched/merge/empty tests and a comparison
  against the free functions; verify with `cargo test -p sciencekit_metrics`.
- [ ] 7.3 Add a concurrency test that partitions a large input across rayon tasks, accumulates
  in parallel, merges and asserts equality with the sequential result for both accumulators,
  plus a compile-time `Send + Sync` assertion; verify with `cargo test -p sciencekit_metrics`.

## 8. Acceptance and integration

- [ ] 8.1 Add `crates/sciencekit_metrics/src/acceptance_tests/` covering tiny data (one
  observation), large data (at least 100 000 rows), every averaging mode, the NaN-aware
  distance on a hand-computed matrix and concurrent accumulation, asserting documented
  tolerances; verify with `cargo test -p sciencekit_metrics`.
- [ ] 8.2 Verify the workspace gates `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` are
  green before any push.

## 9. Documentation, changelog and review

- [ ] 9.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased → Added`
  for `sciencekit_metrics`, the two pairwise kernels and the three error variants; verify the
  entry lists every public artifact shipped.
- [ ] 9.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): update `docs/src/algorithms.md` (metrics status), add a
  `Development Explained` chapter covering metric semantics, label spaces, averaging,
  accumulators and the NaN-aware distance, register it in `SUMMARY.md`, update
  `maintaining.md`, and verify with `mdbook build docs`.
- [ ] 9.3 Independent review by a separate agent validating every scenario in
  `evaluation-metrics`, `pairwise-distances`, `error-model` and `backend-selection` against
  the code and tests; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR (Y-statement) in English and reference it from
  both PRs with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
