# Tasks

## 1. Shared harness and conventions

- [x] 1.1 Add the shared export round-trip test helper and document the dotted-name scheme;
  verify the helper with a mock model.
- [x] 1.2 Add a metadata-version constant and a test that every family uses it; verify with
  `cargo test --workspace export_metadata`.

## 2. Preprocessing and impute

- [x] 2.1 Implement export/import for `SKStandardScaler`, `SKMinMaxScaler` and
  `SKRobustScaler` (`standard_scaler.*`, `minmax_scaler.*`, `robust_scaler.*`) and
  round-trip predictions; verify with `cargo test -p sciencekit_preprocessing export`.
- [x] 2.2 Implement export/import for `SKOneHotEncoder` and `SKPolynomialFeatures`,
  including category/label tables; verify with
  `cargo test -p sciencekit_preprocessing export`.
- [x] 2.3 Implement export/import for `SKSimpleImputer` and `SKMissingIndicator`, including
  indicator scope and strategy metadata; verify with
  `cargo test -p sciencekit_impute export`.

## 3. Linear models

- [x] 3.1 Implement export/import for `SKLinearRegression` and `SKRidge` (coefficients,
  intercepts, alpha dimension, rank/singular values); verify with
  `cargo test -p sciencekit_linear_model export`.
- [x] 3.2 Implement export/import for `SKLasso`, `SKElasticNet` and
  `SKLogisticRegression` (per-alpha solutions, class tables, solver metadata); verify with
  `cargo test -p sciencekit_linear_model export`.
- [x] 3.3 Implement export/import for `SKSGDClassifier`/`SKSGDRegressor` (averaged iterate,
  schedule/penalty metadata, class table); verify with
  `cargo test -p sciencekit_linear_model export`.

## 4. Neighbours, SVM and trees

- [x] 4.1 Implement export/import for KNN and KNNImputer (training design, tree/index
  metadata, weights/metric); verify with `cargo test -p sciencekit_neighbors export` and
  `cargo test -p sciencekit_impute export`.
- [x] 4.2 Implement export/import for SVC/SVR/LinearSVC (support vectors, dual
  coefficients, kernel config, class tables); verify with
  `cargo test -p sciencekit_svm export`.
- [x] 4.3 Implement export/import for decision trees and the ensembles (node arrays, leaf
  values, tree list, class tables, OOB metadata); verify with
  `cargo test -p sciencekit_tree export` and `cargo test -p sciencekit_ensemble export`.

## 5. Clustering, decomposition and outliers

- [x] 5.1 Implement export/import for KMeans, MiniBatchKMeans and DBSCAN (centers, counts,
  labels/core indices, typed assignment metadata); verify with
  `cargo test -p sciencekit_cluster export`.
- [x] 5.2 Implement export/import for PCA, TruncatedSVD and IncrementalPCA (components,
  mean, variance, solver metadata); verify with
  `cargo test -p sciencekit_decomposition export`.
- [ ] 5.3 Implement export/import for IsolationForest (tree arrays, offsets,
  contamination metadata); verify with `cargo test -p sciencekit_outlier export`.

## 6. Pipelines and model selection

- [ ] 6.1 Implement pipeline export/import with stage-qualified names and the final model;
  verify with `cargo test -p sciencekit_pipeline export`.
- [ ] 6.2 Implement grid-search/halving export of the best model and selection metadata;
  verify with `cargo test -p sciencekit_model_selection export`.
- [ ] 6.3 Add the workspace-wide export coverage test and concurrent-export determinism
  test; verify with `cargo test --workspace export_coverage`.

## 7. Documentation and review

- [x] 7.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for model export coverage.
- [x] 7.2 Add a `Development Explained` chapter on stable tensor names, metadata and
  round-trip import; verify with `mdbook build docs`.
- [x] 7.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 7.4 Independent review by a separate agent validating every scenario in
  `model-export-coverage`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.

## Status note

Tasks 1.1–4.2 and 7.1–7.3 are complete on the code, OpenSpec and documentation
branches. Task 4.3 covers decision trees, random forests and gradient boosting
(bagging was already exported); histogram gradient boosting is still pending, so
4.3 stays open. Tasks 5.x (clustering, decomposition, outlier detection), 6.x
(pipelines, model selection, workspace coverage) and 7.4 (independent review)
remain open.
