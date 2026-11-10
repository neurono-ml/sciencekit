# Tasks

## 1. Medium and stage adapter

- [x] 1.1 Create `crates/sciencekit_pipeline` with `Cargo.toml`, `lib.rs`,
  `SKFeatureBlock` re-export and the `From<Array2>`/`From<CsMat>` adapters; verify
  `cargo test -p sciencekit_pipeline` compiles.
- [x] 1.2 Add `From<SKImputedOutput>`/`From<SKIndicatorOutput>` adapters in the impute crate
  and tests; verify with `cargo test -p sciencekit_impute pipeline_adapter`.
- [x] 1.3 Implement the blanket `SKPipelineTransformer` stage adapter and a hand-checkable
  scaler stage test; verify with `cargo test -p sciencekit_pipeline stage`.

## 2. Typed builder

- [x] 2.1 Implement the nested-tuple builder, `then`, and the classifier/regressor
  finalizers; verify a two-stage chain compiles and fits; verify with
  `cargo test -p sciencekit_pipeline builder`.
- [x] 2.2 Add compile-fail doctests for a scalar mismatch and an incompatible final stage;
  verify the harness fails to compile the bad examples and passes the good ones.
- [x] 2.3 Implement the fitted pipeline model with stage and final-model accessors; verify
  with `cargo test -p sciencekit_pipeline model`.

## 3. Fit and prediction

- [x] 3.1 Implement `fit` mapping the stage tuple forward and fitting the final estimator;
  verify a hand-computed scaler→imputer→classifier chain; verify with
  `cargo test -p sciencekit_pipeline fit`.
- [x] 3.2 Implement classifier `predict_labels`/`predict_probabilities` and regressor
  `predict` delegation; verify with `cargo test -p sciencekit_pipeline predict`.
- [x] 3.3 Add dense/sparse flow tests, a representation-changing stage and no-implicit-
  densification checks; verify with `cargo test -p sciencekit_pipeline representation`.
- [x] 3.4 Add repeated-fit determinism, concurrent prediction and `Send + Sync` tests;
  verify with `cargo test -p sciencekit_pipeline concurrency`.

## 4. Acceptance and review

- [x] 4.1 Add `acceptance_tests/` covering tiny and large data, classifier/regressor,
  nested pipelines, dense/sparse, every existing transformer and metric production; verify
  with `cargo test -p sciencekit_pipeline`.
- [x] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on typed stage composition and the block medium); verify
  with `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `typed-pipeline`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
