# Tasks

## 1. Crate and index splits

- [x] 1.1 Create `crates/sciencekit_model_selection` with `Cargo.toml`, `lib.rs`,
  `SKIndexSplit` and the `SKSplitter` contract; verify `cargo test -p
  sciencekit_model_selection` compiles.
- [x] 1.2 Implement `sk_gather_rows` for dense/sparse features and targets, with dense and
  sparse gathering tests (no densification); verify with
  `cargo test -p sciencekit_model_selection gather`.
- [x] 1.3 Implement seeded Fisher-Yates shuffling through the shared generator and test
  reproducibility; verify with `cargo test -p sciencekit_model_selection shuffle`.

## 2. Splitters

- [x] 2.1 Implement `SKKFoldBuilder` and `SKKFold` with disjoint, exhaustive folds and
  seeded shuffling; verify with `cargo test -p sciencekit_model_selection kfold`.
- [x] 2.2 Implement `SKStratifiedKFold` with class proportion preservation and the
  rare-class rejection; verify with
  `cargo test -p sciencekit_model_selection stratified`.
- [x] 2.3 Implement `SKGroupKFold` with group integrity, default group count and length
  validation; verify with `cargo test -p sciencekit_model_selection group`.

## 3. Train/test split and cross-validation

- [x] 3.1 Implement `sk_train_test_split` with fractions/counts, shuffle, seeded order and
  optional stratification; verify with
  `cargo test -p sciencekit_model_selection train_test_split`.
- [x] 3.2 Implement `sk_cross_val_score` for classifier and regressor policies, with one
  score per fold and parallel-equals-sequential; verify with
  `cargo test -p sciencekit_model_selection cross_validation`.
- [x] 3.3 Add validation tests for empty input, invalid fractions, too many splits,
  class/group cardinality, length mismatch and non-finite values; verify with
  `cargo test -p sciencekit_model_selection validation`.

## 4. Acceptance and review

- [x] 4.1 Add `acceptance_tests/` covering tiny and large data, dense/sparse, all
  splitters, stratification/group properties, determinism, concurrency and metric
  production; verify with `cargo test -p sciencekit_model_selection`.
- [x] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on index splits, stratification and grouped folds);
  verify with `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `model-selection-splits`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
