# Tasks

## 1. Configuration and builder

- [x] 1.1 Create `crates/sciencekit_svm/src/linear_svc/` with dispatcher `mod.rs`,
  `config.rs` and `builder.rs`; implement the defaults (`C = 1.0`, `SquaredHinge`,
  `None`, intercept on, tolerance, iteration budget, `Faer`, `Automatic`) and
  invalid-value tests; verify with `cargo test -p sciencekit_svm linear_svc_builder`.
- [x] 1.2 Implement `SKLinearSvcLoss` (`Hinge`, `SquaredHinge`) and `SKClassWeight`
  resolution, with tests for default loss, explicit loss and explicit/balanced weights;
  verify with `cargo test -p sciencekit_svm linear_svc_config`.

## 2. Design abstraction and solver

- [x] 2.1 Write tests for the dense design row dot and scaled-row accumulation on a
  hand-computed matrix, confirm the failure, then implement `design.rs` for owned dense
  data; verify with `cargo test -p sciencekit_svm linear_svc_design`.
- [x] 2.2 Add the borrowed CSR design path with tests proving stored-entry-only row dots
  and no densification; verify with `cargo test -p sciencekit_svm linear_svc_design`.
- [x] 2.3 Write tests for the dual coordinate update on a two-point problem and for the
  projected gradient stopping rule, then implement `solver.rs` for the L1-hinge loss;
  verify with `cargo test -p sciencekit_svm linear_svc_solver`.
- [x] 2.4 Add the L2-squared-hinge loss path (`U = ∞`, diagonal shift `1/(2C)`) and a
  comparison test proving both losses converge on the same separable fixture; verify with
  `cargo test -p sciencekit_svm linear_svc_solver`.
- [x] 2.5 Add the iteration budget and `NotConverged { iterations }` test; verify with
  `cargo test -p sciencekit_svm linear_svc_solver`.

## 3. Estimator

- [x] 3.1 Implement binary `fit` producing a model with one coefficient vector and one
  intercept, and verify the separable and overlapping fixtures; verify with
  `cargo test -p sciencekit_svm linear_svc_fit`.
- [x] 3.2 Implement one-vs-rest multiclass fitting with parallel heads and deterministic
  head order; verify a three-class problem retains three heads; verify with
  `cargo test -p sciencekit_svm linear_svc_multiclass`.
- [x] 3.3 Implement `decision_function` and `predict_labels` with canonical tie-breaking;
  verify shape, argmax consistency and a hand-built tie; verify with
  `cargo test -p sciencekit_svm linear_svc_predict`.
- [x] 3.4 Implement `predict_probabilities` returning `UnsupportedOperation` with
  guidance; verify with `cargo test -p sciencekit_svm linear_svc_probability`.
- [x] 3.5 Add class-weight tests proving `Balanced` changes the boundary on an imbalanced
  fixture and explicit weights are honored; verify with
  `cargo test -p sciencekit_svm linear_svc_weights`.

## 4. Representation parity and validation

- [x] 4.1 Add dense-vs-sparse fit equivalence and sparse prediction tests, asserting
  coefficients/predictions agree and no sparse input is densified; verify with
  `cargo test -p sciencekit_svm linear_svc_representation`.
- [x] 4.2 Add tests for empty input, single-class targets, feature-count mismatch,
  non-finite values, non-positive `C`, zero tolerance/iteration budget and unsupported
  loss, asserting the exact central error variant; verify with
  `cargo test -p sciencekit_svm linear_svc_validation`.
- [x] 4.3 Add repeated-fit determinism and concurrent-prediction tests, plus a
  `Send + Sync` assertion; verify with `cargo test -p sciencekit_svm linear_svc_concurrency`.

## 5. Acceptance

- [x] 5.1 Add `acceptance_tests/` covering tiny data, large data (at least 20 000 rows and
  10 000 features in a sparse fixture), both losses, multiclass, class weights, dense and
  sparse paths, parallel-equals-sequential prediction and metric production through
  `sciencekit_metrics`; verify with `cargo test -p sciencekit_svm`.
- [ ] 5.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.

## 6. Documentation, changelog and review

- [x] 6.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for `SKLinearSVC`, the dual coordinate-descent solver and the one-vs-rest
  multiclass path.
- [x] 6.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the linear SVM chapter, register it in `SUMMARY.md`, add a
  `Development Explained` chapter covering the dual formulation, coordinate descent,
  sparse row dots and one-vs-rest, update `maintaining.md`, and verify with
  `mdbook build docs`.
- [x] 6.3 Record the deferred `memmap2`/export work and the deliberate absence of
  probabilities in `docs/PRD.md`; verify the amended paragraphs render.
- [x] 6.4 Independent review by a separate agent validating every scenario in
  `linear-support-vector-classifier` against the code and tests; address every finding
  and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
