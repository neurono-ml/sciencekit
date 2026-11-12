# Tasks

## 1. Scaffold and binned dataset

- [x] 1.1 Add `bitvec` to the workspace dependencies and create the
  `histogram_boosting` module scaffold (`mod.rs`, `binning`, `histogram`, `growth`);
  verify `cargo test -p sciencekit_ensemble` compiles.
- [x] 1.2 Write and implement quantile edge derivation (at most `max_bins`, deterministic
  subsampling for large columns), value-to-bin assignment and the dedicated missing bin;
  verify with `cargo test -p sciencekit_ensemble hist_binning`.
- [x] 1.3 Implement the column-major `SKBinnedDataset` store (`u8` bins), category mapping
  and unknown-category bin, with tests for dimensions, layout and no densification of
  sparse input; verify with `cargo test -p sciencekit_ensemble hist_dataset`.

## 2. Histogram split finding

- [x] 2.1 Implement per-feature gradient/Hessian/count accumulation over a node's sample
  indices, parallel across features, with a hand-computed test; verify with
  `cargo test -p sciencekit_ensemble hist_accumulate`.
- [x] 2.2 Implement the numeric prefix scan and the gain formula with
  `minimum_samples_leaf` and `l2_regularization`, with a hand-computed best split; verify
  with `cargo test -p sciencekit_ensemble hist_gain`.
- [x] 2.3 Implement histogram subtraction from the parent and a test comparing subtraction
  with direct accumulation; verify with
  `cargo test -p sciencekit_ensemble hist_subtract`.
- [x] 2.4 Implement categorical category ordering by `G/H`, prefix partitions and the
  `BitVec` membership mask, with a hand-computed partition test; verify with
  `cargo test -p sciencekit_ensemble hist_categorical`.

## 3. Growth and missing routing

- [x] 3.1 Implement leaf-wise growth with a gain priority queue, `max_leaf_nodes`,
  `minimum_samples_leaf` and the flat `SKTreeNode` array; verify a hand-computed growth
  sequence; verify with `cargo test -p sciencekit_ensemble hist_growth`.
- [x] 3.2 Implement missing-bin routing and the learned `default_left` direction, with
  tests for a missing row going left/right and a node with no missing rows; verify with
  `cargo test -p sciencekit_ensemble hist_missing`.

## 4. Boosting and estimators

- [x] 4.1 Wire the W4.4 loss/gradient/Hessian contract, initial prediction, learning rate
  and Newton leaves into the histogram stage loop; verify a single-stage hand-computed
  update; verify with `cargo test -p sciencekit_ensemble hist_stage`.
- [x] 4.2 Implement `SKHistGradientBoostingClassifier` builder, binary/multiclass fit and
  prediction (softmax/sigmoid), with defaults and invalid-value tests; verify with
  `cargo test -p sciencekit_ensemble hist_classifier`.
- [x] 4.3 Implement `SKHistGradientBoostingRegressor` builder, fit and prediction for
  squared and absolute error; verify with `cargo test -p sciencekit_ensemble hist_regressor`.
- [x] 4.4 Implement early stopping with the seeded validation split and fitted-iteration
  reporting; verify with `cargo test -p sciencekit_ensemble hist_early_stop`.

## 5. Representation and validation

- [ ] 5.1 Implement raw dense and sparse prediction binning and traversal, with
  dense-vs-sparse equivalence and no-densification tests; verify with
  `cargo test -p sciencekit_ensemble hist_representation`.
- [ ] 5.2 Add validation tests for empty input, invalid `max_bins`/leaf/learning
  parameters, unknown categorical columns, single-class classification, non-finite values,
  feature-count mismatch and invalid validation fraction; verify with
  `cargo test -p sciencekit_ensemble hist_validation`.
- [ ] 5.3 Add repeated-fit determinism, thread-count invariance and concurrent prediction
  tests, plus a `Send + Sync` assertion; verify with
  `cargo test -p sciencekit_ensemble hist_determinism`.

## 6. Acceptance

- [ ] 6.1 Add `acceptance_tests/` covering tiny and large data (at least 50 000 rows),
  binary/multiclass classification, regression, categorical features, missing values,
  early stopping, dense/sparse and metric production through `sciencekit_metrics`; verify
  with `cargo test -p sciencekit_ensemble`.

## 7. Documentation, changelog and review

- [ ] 7.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the histogram boosters, binning, categorical partitions and missing routing.
- [ ] 7.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the histogram-boosting chapter, register it in `SUMMARY.md`,
  add a `Development Explained` chapter covering binning, histograms, leaf-wise growth,
  categorical bitmasks and missing routing, update `maintaining.md`, and verify with
  `mdbook build docs`.
- [ ] 7.3 Record the deferred `memmap2` streaming store, constraints and export in
  `docs/PRD.md`; verify the amended paragraphs render.
- [ ] 7.4 Independent review by a separate agent validating every scenario in
  `histogram-gradient-boosting` against the code and tests; address every finding and
  re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
