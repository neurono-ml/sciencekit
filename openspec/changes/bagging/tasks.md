# Tasks

## 1. Bagging module and error type

- [x] 1.1 Create `crates/sciencekit_ensemble/src/bagging/` with dispatcher `mod.rs`,
  `error.rs` and `config.rs`; define `SKBaggingError<E>` with `From<SKError>` and
  `From<E>` and test both conversions; verify with
  `cargo test -p sciencekit_ensemble bagging_error`.
- [x] 1.2 Implement the generic builder parameters (`number_of_estimators`,
  `maximum_samples`, `maximum_features`, `bootstrap`, `bootstrap_features`, `oob_score`,
  `random_seed`, `backend`, `execution_mode`) with default and invalid-value tests; verify
  with `cargo test -p sciencekit_ensemble bagging_builder`.

## 2. Generic fitting contract

- [x] 2.1 Write compile-time tests proving a classifier base configuration satisfies
  `SKBaggingClassifier<B>` and a regressor configuration does not; verify with
  `cargo test -p sciencekit_ensemble bagging_bounds`.
- [x] 2.2 Implement bootstrap row sampling reusing the shared sampler and, when enabled,
  feature subset sampling; verify row/feature reproducibility and sample shapes; verify
  with `cargo test -p sciencekit_ensemble bagging_sampling`.
- [x] 2.3 Implement parallel per-estimator fitting (each estimator on its row and feature
  subset) with deterministic seeds and lock-free slots; verify parallel-equals-sequential
  and repeated-fit determinism; verify with
  `cargo test -p sciencekit_ensemble bagging_fit`.

## 3. Prediction and routing

- [x] 3.1 Implement feature-subset routing for prediction and a test that an estimator
  fitted on features `[0, 2]` receives only those columns; verify with
  `cargo test -p sciencekit_ensemble bagging_routing`.
- [x] 3.2 Implement classifier probability averaging and label prediction with canonical
  tie-breaking; verify a hand-computed ensemble; verify with
  `cargo test -p sciencekit_ensemble bagging_classifier`.
- [x] 3.3 Implement regressor response averaging; verify a hand-computed ensemble; verify
  with `cargo test -p sciencekit_ensemble bagging_regressor`.

## 4. OOB, representation and validation

- [x] 4.1 Implement OOB scoring from the shared masks and test accuracy/R², no-scored-rows
  and the disabled case; verify with `cargo test -p sciencekit_ensemble bagging_oob`.
- [x] 4.2 Add representation tests with a sparse-capable base estimator and a dense-only
  base estimator, asserting the base error surfaces through `SKBaggingError::Base`; verify
  with `cargo test -p sciencekit_ensemble bagging_representation`.
- [x] 4.3 Add validation tests for empty input, zero estimator count, invalid fractions,
  feature-count mismatch and non-finite values; verify with
  `cargo test -p sciencekit_ensemble bagging_validation`.
- [x] 4.4 Add a `Send + Sync` assertion and a concurrent-prediction test sharing one
  bagging model; verify with `cargo test -p sciencekit_ensemble bagging_concurrency`.

## 5. Acceptance

- [x] 5.1 Add `acceptance_tests/` covering tiny and large data (at least 10 000 rows),
  tree and non-tree base estimators, feature bootstrap, OOB on/off, dense/sparse, thread-
  count invariance and metric production through `sciencekit_metrics`; verify with
  `cargo test -p sciencekit_ensemble`.

## 6. Documentation, changelog and review

- [ ] 6.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the generic bagging estimators, feature routing and OOB scoring.
- [ ] 6.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the bagging chapter, register it in `SUMMARY.md`, add a
  `Development Explained` chapter covering the generic base-estimator contract, feature
  routing and OOB, update `maintaining.md`, and verify with `mdbook build docs`.
- [ ] 6.3 Record the heterogeneous-ensemble and streaming/memmap/export non-goals in
  `docs/PRD.md`; verify the amended paragraphs render.
- [ ] 6.4 Independent review by a separate agent validating every scenario in
  `bagging-ensembles` against the code and tests; address every finding and re-run the
  gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
