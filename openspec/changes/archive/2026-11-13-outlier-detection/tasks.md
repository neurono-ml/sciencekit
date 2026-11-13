# Tasks

## 1. Contract and crate

- [x] 1.1 Add `SKOutlierLabel { Inlier, Outlier }` with its one-byte size assertion and
  `to_i64`/`from_i64` conversions, plus `SKOutlierPredictor<F>` (`predict`,
  `decision_function`, `score_samples`), with a constant detector test; re-export both and
  verify with `cargo test -p sciencekit_common outlier`.
- [x] 1.2 Create `crates/sciencekit_outlier` with `Cargo.toml`, `lib.rs` and the
  `isolation_forest` dispatcher; add it to `[workspace.members]`; verify
  `cargo test -p sciencekit_outlier` compiles.

## 2. Isolation trees

- [x] 2.1 Implement the random-feature/random-threshold split and growth limit, with tests
  for a one-sample leaf and a threshold inside the observed range; verify with
  `cargo test -p sciencekit_outlier isolation_tree`.
- [x] 2.2 Implement path length (`c(n)`, leaf size) and the expected-depth reduction, with
  hand-computed tests; verify with `cargo test -p sciencekit_outlier isolation_path`.
- [x] 2.3 Implement dense and sparse tree building and dense/sparse score equivalence
  without densification; verify with
  `cargo test -p sciencekit_outlier isolation_representation`.

## 3. Estimator and scoring

- [x] 3.1 Implement the builder with the documented defaults and invalid-value tests;
  verify with `cargo test -p sciencekit_outlier isolation_builder`.
- [x] 3.2 Implement fit (subsample, per-tree seeds, forest assembly) and the model state;
  verify with `cargo test -p sciencekit_outlier isolation_fit`.
- [x] 3.3 Implement `score_samples`, `decision_function`, typed `predict` and `fit_predict`
  with the contamination offset; verify with
  `cargo test -p sciencekit_outlier isolation_score`.
- [x] 3.4 Add validation tests for empty input, zero trees, invalid sample size,
  contamination range, feature-count mismatch and non-finite values; verify with
  `cargo test -p sciencekit_outlier isolation_validation`.
- [x] 3.5 Add repeated-fit determinism, thread-count invariance, concurrent prediction and
  `Send + Sync` tests; verify with
  `cargo test -p sciencekit_outlier isolation_concurrency`.

## 4. Acceptance and review

- [x] 4.1 Add `acceptance_tests/` covering tiny data, large data (at least 20 000 rows),
  dense/sparse, contamination Auto/fraction, determinism, concurrency and metric
  production; verify with `cargo test -p sciencekit_outlier`.
- [x] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on isolation trees, path length and the contamination
  offset); verify with `mdbook build docs`.
- [x] 4.4 Record the LOF/One-Class SVM/missing-value/streaming/GPU/memmap/export non-goals
  in `docs/PRD.md`.
- [x] 4.5 Independent review by a separate agent validating every scenario in
  `outlier-prediction` and `isolation-forest`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
