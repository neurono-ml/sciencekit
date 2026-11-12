# Tasks

## 1. Tree advanced surface

- [x] 1.1 Write `sciencekit_tree/structure_tests.rs` for a structure-only build on a
  response vector, leaf-count reporting and leaf-value replacement; confirm the failures,
  then implement `structure/build.rs` and `structure/leaf_values.rs`; verify with
  `cargo test -p sciencekit_tree structure`.
- [x] 1.2 Add a test proving `SKDecisionTreeRegressor` uses its public criterion and is
  unaffected by the advanced surface; verify with
  `cargo test -p sciencekit_tree structure`.

## 2. Loss contract

- [x] 2.1 Implement `SKGradientLoss` with the initial prediction, negative gradient,
  Hessian, link and loss methods, and the squared-error implementation; verify hand-computed
  gradients, Hessians and initial mean; verify with
  `cargo test -p sciencekit_ensemble boosting_loss`.
- [x] 2.2 Implement absolute-error and Huber losses with their documented degeneracy
  (constant Hessian for absolute error, clipped gradient for Huber) and hand-computed
  tests; verify with `cargo test -p sciencekit_ensemble boosting_loss`.
- [x] 2.3 Implement binary log-loss with sigmoid and multiclass log-loss with softmax,
  including initial log-odds/log-priors and hand-computed gradients/Hessians; verify with
  `cargo test -p sciencekit_ensemble boosting_log_loss`.

## 3. Boosting substrate

- [x] 3.1 Implement the stage loop (gradient computation, structure build, Newton leaf
  replacement, shrinkage, raw-score update) and test a single-stage hand-computed update;
  verify with `cargo test -p sciencekit_ensemble boosting_stage`.
- [x] 3.2 Implement seeded row subsampling and feature subsampling, with reproducibility
  and full-sample tests; verify with
  `cargo test -p sciencekit_ensemble boosting_subsample`.
- [x] 3.3 Implement early stopping with the seeded validation split, best-stage restore and
  stage-count reporting; verify with `cargo test -p sciencekit_ensemble boosting_early_stop`.

## 4. Classifier

- [x] 4.1 Implement `SKGradientBoostingClassifierBuilder` with the documented defaults and
  invalid-value tests; verify with
  `cargo test -p sciencekit_ensemble gb_classifier_builder`.
- [x] 4.2 Implement binary fit and probability/label prediction; verify a hand-computed
  score and a separable fixture; verify with
  `cargo test -p sciencekit_ensemble gb_classifier`.
- [x] 4.3 Implement multiclass fit (one tree per class per stage), softmax probabilities
  and canonical labels; verify with `cargo test -p sciencekit_ensemble gb_classifier`.

## 5. Regressor

- [x] 5.1 Implement `SKGradientBoostingRegressorBuilder` with the documented defaults and
  invalid-value tests; verify with
  `cargo test -p sciencekit_ensemble gb_regressor_builder`.
- [x] 5.2 Implement fit and prediction for squared error, absolute error and Huber; verify
  hand-computed updates and a smooth response; verify with
  `cargo test -p sciencekit_ensemble gb_regressor`.

## 6. Importances, representation and validation

- [x] 6.1 Implement aggregated feature importances and test normalization across stages;
  verify with `cargo test -p sciencekit_ensemble gb_importances`.
- [x] 6.2 Add dense-vs-sparse fit/predict equivalence tests and a no-densification test;
  verify with `cargo test -p sciencekit_ensemble gb_representation`.
- [x] 6.3 Add validation tests for empty input, non-positive learning rate/estimator
  count, invalid subsample/feature/regularization, invalid validation fraction,
  single-class classification and non-finite values; verify with
  `cargo test -p sciencekit_ensemble gb_validation`.
- [x] 6.4 Add repeated-fit determinism, thread-count invariance and concurrent prediction
  tests, plus a `Send + Sync` assertion; verify with
  `cargo test -p sciencekit_ensemble gb_determinism`.

## 7. Acceptance

- [x] 7.1 Add `acceptance_tests/` covering tiny and large data (at least 20 000 rows),
  binary/multiclass classification, all regression losses, subsampling, early stopping,
  dense/sparse and metric production through `sciencekit_metrics`; verify with
  `cargo test -p sciencekit_ensemble`.

## 8. Documentation, changelog and review

- [x] 8.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the gradient-boosting estimators, the loss contract and the tree structure
  surface.
- [x] 8.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the gradient-boosting chapter, register it in `SUMMARY.md`,
  add a `Development Explained` chapter covering gradients, Hessians, Newton leaves,
  shrinkage and early stopping, update `maintaining.md`, and verify with `mdbook build
  docs`.
- [x] 8.3 Record the histogram/categorical/missing-value/streaming/memmap/export
  non-goals in `docs/PRD.md`; verify the amended paragraphs render.
- [x] 8.4 Independent review by a separate agent validating every scenario in
  `gradient-boosting` and the `decision-trees` delta against the code and tests; address
  every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
