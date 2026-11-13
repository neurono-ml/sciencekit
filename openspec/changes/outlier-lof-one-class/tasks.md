# Tasks

Every task follows the red-green-refactor loop: write the failing companion test first,
confirm the failure, implement the minimum that passes and refactor with the gates green.

## 1. LOF geometry

- [ ] 1.1 Add `SKLocalOutlierFactorBuilder` with the documented defaults and
  invalid-value tests for the neighbour count, contamination, metric and backend; verify
  with `cargo test -p sciencekit_outlier lof_builder`.
- [ ] 1.2 Implement core distance, reachability distance and local reachability density
  with hand-computed two-point and cluster fixtures, including the duplicate-point rule;
  verify with `cargo test -p sciencekit_outlier lof_geometry`.
- [ ] 1.3 Implement the local outlier factor and negative outlier factor with
  uniform-density, isolated-point and finite-value tests; verify with
  `cargo test -p sciencekit_outlier lof_factor`.
- [ ] 1.4 Implement the dense fit path with the KD-tree searcher, the contamination offset,
  `fit_predict` and the negative outlier factor vector; verify with
  `cargo test -p sciencekit_outlier lof_fit`.

## 2. LOF novelty, representation and exports

- [ ] 2.1 Implement the fit-only typed refusal and the novelty prediction path with
  new-row neighbour, density and factor tests; verify with
  `cargo test -p sciencekit_outlier lof_novelty`.
- [ ] 2.2 Implement the sparse and mixed-representation branch through the brute-force
  searcher with dense/sparse equivalence and non-densification tests; verify with
  `cargo test -p sciencekit_outlier lof_representation`.
- [ ] 2.3 Implement the typed `predict`, `decision_function` and `score_samples` outputs
  with sign, offset and batch-shape tests; verify with
  `cargo test -p sciencekit_outlier lof_predict`.
- [ ] 2.4 Implement `SKExportable`/`SKImportable` for the LOF model with dense and CSR
  round-trip, metadata and prediction-parity tests; verify with
  `cargo test -p sciencekit_outlier lof_export`.

## 3. One-Class SVM

- [ ] 3.1 Add `SKOneClassSVMBuilder` with the documented defaults and `nu`, kernel,
  degree, gamma and `coef0` validation tests; verify with
  `cargo test -p sciencekit_outlier one_class_builder`.
- [ ] 3.2 Formulate and solve the one-class dual with the shared SMO primitives, with
  bounded-coefficient, unit-sum and convergence tests on two-point fixtures; verify with
  `cargo test -p sciencekit_outlier one_class_solver`.
- [ ] 3.3 Implement support-vector extraction, `rho` and the decision expansion with
  hand-computed kernel fixtures; verify with
  `cargo test -p sciencekit_outlier one_class_decision`.
- [ ] 3.4 Implement the four supported kernels end to end with per-kernel fit, catalogue
  parity and invalid-parameter tests; verify with
  `cargo test -p sciencekit_outlier one_class_kernels`.
- [ ] 3.5 Implement dense, sparse and mixed kernel/matrix paths with dense/sparse
  equivalence and non-densification tests; verify with
  `cargo test -p sciencekit_outlier one_class_representation`.
- [ ] 3.6 Implement the typed `predict`, `decision_function` and `score_samples` outputs
  and `fit_predict`; verify with
  `cargo test -p sciencekit_outlier one_class_predict`.
- [ ] 3.7 Implement `SKExportable`/`SKImportable` for the One-Class SVM model with dense and
  CSR round-trip and prediction-parity tests; verify with
  `cargo test -p sciencekit_outlier one_class_export`.

## 4. Validation, acceptance and review

- [ ] 4.1 Add validation tests for empty input, invalid neighbour count, `nu` range,
  feature-count mismatch, non-finite values and prediction before fit; verify with
  `cargo test -p sciencekit_outlier lof_one_class_validation`.
- [ ] 4.2 Add determinism, thread-count invariance, concurrent prediction and `Send + Sync`
  tests for both estimators; verify with
  `cargo test -p sciencekit_outlier outlier_concurrency`.
- [ ] 4.3 Add `acceptance_tests/` covering tiny data, large data (at least 20 000 rows),
  dense/sparse, contamination `Auto`/fraction, both novelty modes, every kernel and metric
  production; verify with `cargo test -p sciencekit_outlier`.
- [ ] 4.4 Add observability tests asserting the configured backend kind on every span of
  `fit`, `fit_predict`, `predict`, `decision_function` and `score_samples`; verify with
  `cargo test -p sciencekit_outlier outlier_backend_spans`.
- [ ] 4.5 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.6 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on reachability distance, local density, one-class SMO
  and `nu`); verify with `mdbook build docs`.
- [ ] 4.7 Independent review by a separate agent validating every scenario in
  `outlier-lof-one-class`, the data-view boundary decision and the export contract;
  address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
