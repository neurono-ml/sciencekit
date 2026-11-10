# Tasks

## 1. Crate scaffold and tree structure

- [x] 1.1 Create `crates/sciencekit_tree` with `Cargo.toml`, `lib.rs`, the `node` module
  and the classifier/regressor dispatcher `mod.rs` files; add it to
  `[workspace.members]`; verify `cargo test -p sciencekit_tree` compiles.
- [x] 1.2 Write and implement `SKTreeNode` accessors (leaf sentinel, split feature,
  threshold, children, sample count, impurity, value) and a tiny traversal helper, with
  tests for a hand-built three-node tree; verify with
  `cargo test -p sciencekit_tree node`.

## 2. Impurity and split search

- [x] 2.1 Write and implement Gini and entropy impurity over class-count vectors, with
  hand-computed tests for pure and mixed nodes; verify with
  `cargo test -p sciencekit_tree impurity`.
- [x] 2.2 Write and implement squared-error and absolute-error impurity over response
  vectors, with hand-computed tests for constant and mixed nodes; verify with
  `cargo test -p sciencekit_tree impurity`.
- [x] 2.3 Implement the dense design split scan: sort the node's sample indices by a
  feature, accumulate criterion statistics, evaluate valid thresholds and return the best
  candidate; verify with a hand-computed two-feature fixture; verify with
  `cargo test -p sciencekit_tree splitter_dense`.
- [x] 2.4 Implement the sparse CSC design built once at fit and the sparse split scan
  merging stored entries with implicit zeros; verify dense-vs-sparse split equivalence
  and that no dense copy of the design is created; verify with
  `cargo test -p sciencekit_tree splitter_sparse`.
- [x] 2.5 Implement `Splitter::Best` feature-parallel evaluation and deterministic
  tie-breaking (smaller feature, then smaller threshold); verify with
  `cargo test -p sciencekit_tree splitter_best`.
- [x] 2.6 Implement `Splitter::Random` with the seeded generator and test reproducibility;
  verify with `cargo test -p sciencekit_tree splitter_random`.

## 3. Growth constraints and feature subsampling

- [x] 3.1 Enforce `maximum_depth`, `minimum_samples_split` and `minimum_samples_leaf` in
  growth, with tests proving each constraint stops a split; verify with
  `cargo test -p sciencekit_tree growth`.
- [x] 3.2 Implement `maximum_features` (`All`, `Count`, `Fraction`, `Sqrt`, `Log2`) with
  seeded selection and tests for the resolved feature count; verify with
  `cargo test -p sciencekit_tree feature_subsampling`.

## 4. Classifier

- [x] 4.1 Implement `SKDecisionTreeClassifierBuilder` with the documented defaults and
  invalid-value tests; verify with `cargo test -p sciencekit_tree classifier_builder`.
- [x] 4.2 Implement classifier fit producing the node array, leaf class counts and the
  canonical class table; verify a perfectly separable fixture and a multi-class fixture;
  verify with `cargo test -p sciencekit_tree classifier_fit`.
- [x] 4.3 Implement `predict_labels` and `predict_probabilities` from the leaf counts,
  with majority prediction, probabilities summing to one and deterministic class order;
  verify with `cargo test -p sciencekit_tree classifier_predict`.
- [x] 4.4 Implement class weights and test their effect on an imbalanced fixture; verify
  with `cargo test -p sciencekit_tree classifier_weights`.

## 5. Regressor

- [x] 5.1 Implement `SKDecisionTreeRegressorBuilder` with the documented defaults and
  invalid-value tests; verify with `cargo test -p sciencekit_tree regressor_builder`.
- [x] 5.2 Implement squared-error fit and `predict` with the leaf mean; verify a
  hand-computed fixture; verify with `cargo test -p sciencekit_tree regressor`.
- [x] 5.3 Implement absolute-error fit and `predict` with the leaf median; verify a
  hand-computed fixture; verify with `cargo test -p sciencekit_tree regressor`.

## 6. Model accessors, representation and determinism

- [x] 6.1 Implement feature importances from the impurity decrease and test normalization
  on a separating feature; verify with `cargo test -p sciencekit_tree importances`.
- [x] 6.2 Add dense-vs-sparse fit/predict equivalence tests and a test proving sparse fit
  does not densify; verify with `cargo test -p sciencekit_tree representation`.
- [x] 6.3 Add repeated-fit determinism, seeded `Random` splitter and parallel-equals-
  sequential prediction tests, plus a `Send + Sync` assertion; verify with
  `cargo test -p sciencekit_tree determinism`.

## 7. Validation

- [x] 7.1 Add tests for empty input, non-finite values, feature-count mismatch,
  single-class classification, invalid depth/split/leaf parameters and invalid feature
  subsampling, asserting the exact central error variant; verify with
  `cargo test -p sciencekit_tree validation`.

## 8. Acceptance

- [x] 8.1 Add `acceptance_tests/` covering tiny data (one sample, one split), large data
  (at least 20 000 rows), classification and regression, every criterion, dense and
  sparse paths, depth/leaf constraints, feature subsampling, concurrency and metric
  production through `sciencekit_metrics`; verify with `cargo test -p sciencekit_tree`.
- [x] 8.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.

## 9. Documentation, changelog and review

- [x] 9.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the tree crate, both estimators, the splitter and feature importances.
- [x] 9.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the decision-tree chapter, register it in `SUMMARY.md`, add
  a `Development Explained` chapter covering CART, impurity, sorted split scans,
  constraints and importances, update `maintaining.md`, and verify with `mdbook build
  docs`.
- [x] 9.3 Record the deferred missing-value routing, pruning, `memmap2` backing and
  persistence/export in `docs/PRD.md`; verify the amended paragraphs render.
- [ ] 9.4 Independent review by a separate agent validating every scenario in
  `decision-trees` against the code and tests; address every finding and re-run the
  gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
