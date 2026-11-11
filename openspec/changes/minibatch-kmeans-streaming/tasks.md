# Tasks

## 1. Module and state

- [x] 1.1 Create `crates/sciencekit_cluster/src/minibatch_kmeans/` with dispatcher
  `mod.rs`, config, state and model; verify `cargo test -p sciencekit_cluster minibatch`
  compiles.
- [x] 1.2 Implement the per-center update and count bookkeeping, with the one-center
  closed form and learning-rate decay tests; verify with
  `cargo test -p sciencekit_cluster minibatch_update`.
- [x] 1.3 Implement the deterministic reassignment rule and tests for threshold crossing
  and outlier choice; verify with `cargo test -p sciencekit_cluster minibatch_reassign`.

## 2. Initialization and partial fit

- [x] 2.1 Implement initialization on the bounded sample and the best-of-`n_init` selection;
  verify with `cargo test -p sciencekit_cluster minibatch_init`.
- [x] 2.2 Implement `partial_fit` preserving centers/counts across calls, with a batched-
  equals-full-fit test; verify with `cargo test -p sciencekit_cluster minibatch_partial`.
- [x] 2.3 Implement dense and sparse batch assignment/update parity and no-densification
  tests; verify with `cargo test -p sciencekit_cluster minibatch_representation`.

## 3. Streaming

- [x] 3.1 Implement `fit_streaming` over `SKSourceProvider` through the shared executor, with
  a one-pass test and a reopenable multi-epoch test; verify with
  `cargo test -p sciencekit_cluster minibatch_streaming`.
- [x] 3.2 Implement the positive-tolerance early stop and batch/epoch reporting; verify with
  `cargo test -p sciencekit_cluster minibatch_streaming`.

## 4. Estimator, model and validation

- [x] 4.1 Implement the builder with the documented defaults and invalid-value tests;
  verify with `cargo test -p sciencekit_cluster minibatch_builder`.
- [x] 4.2 Implement `SKUnsupervisedFit`, `SKClustererPredictor`, `transform` and
  `fit_predict`; verify shapes, nearest-center semantics and scalar independence; verify
  with `cargo test -p sciencekit_cluster minibatch_model`.
- [ ] 4.3 Add validation tests for empty/small batches, zero clusters, invalid
  initialization/reassignment/tolerance, non-finite values and feature-count mismatch;
  verify with `cargo test -p sciencekit_cluster minibatch_validation`.
- [ ] 4.4 Add repeated-run determinism, thread-count invariance and concurrent assignment
  tests, plus a `Send + Sync` assertion; verify with
  `cargo test -p sciencekit_cluster minibatch_determinism`.

## 5. Acceptance and review

- [ ] 5.1 Add `acceptance_tests/` covering tiny data, large streaming data (at least
  50 000 rows), dense/sparse, one/multi-epoch providers, comparison with full KMeans and
  metric production; verify with `cargo test -p sciencekit_cluster`.
- [ ] 5.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 5.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on per-center learning rates, reassignment and streaming
  epochs); verify with `mdbook build docs`.
- [ ] 5.4 Record the no-Elkan/GPU/memmap/export non-goals in `docs/PRD.md`.
- [ ] 5.5 Independent review by a separate agent validating every scenario in
  `mini-batch-k-means`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
