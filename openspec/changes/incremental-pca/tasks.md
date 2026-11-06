# Tasks

## 1. Accumulator and merge

- [ ] 1.1 Create `crates/sciencekit_decomposition/src/incremental_pca/` with dispatcher
  `mod.rs`, config and accumulator; verify `cargo test -p sciencekit_decomposition
  incremental` compiles.
- [ ] 1.2 Implement the Chan-Golub-LeVeque merge and test two-batches-equal-one-batch,
  mean correctness and state persistence; verify with
  `cargo test -p sciencekit_decomposition incremental_merge`.
- [ ] 1.3 Implement the sparse Gram-based accumulation and dense-vs-sparse equality without
  densification; verify with `cargo test -p sciencekit_decomposition incremental_sparse`.

## 2. Finalize and estimator

- [ ] 2.1 Implement finalization (covariance, eigh, sorting, components, explained variance
  and ratio) and the equality-with-PCA test; verify with
  `cargo test -p sciencekit_decomposition incremental_finalize`.
- [ ] 2.2 Implement the builder with the documented defaults and invalid-value tests;
  verify with `cargo test -p sciencekit_decomposition incremental_builder`.
- [ ] 2.3 Implement `partial_fit`, `fit` chunking and model accessors; verify with
  `cargo test -p sciencekit_decomposition incremental_fit`.
- [ ] 2.4 Implement `fit_streaming` over `SKSourceProvider` with one-pass and reopenable
  multi-epoch tests; verify with
  `cargo test -p sciencekit_decomposition incremental_streaming`.
- [ ] 2.5 Implement `transform`, `inverse_transform` and whitening for dense and sparse
  views, with shape and unit-variance tests; verify with
  `cargo test -p sciencekit_decomposition incremental_transform`.

## 3. Validation and concurrency

- [ ] 3.1 Add validation tests for empty batches, component count, batch-size restriction,
  feature-count mismatch and non-finite values; verify with
  `cargo test -p sciencekit_decomposition incremental_validation`.
- [ ] 3.2 Add repeated-fit determinism, batch-size invariance, concurrent transform and
  `Send + Sync` tests; verify with
  `cargo test -p sciencekit_decomposition incremental_concurrency`.

## 4. Acceptance and review

- [ ] 4.1 Add `acceptance_tests/` covering tiny and large dense/sparse data, one/multi-epoch
  streaming, batch-size variation, whitening, determinism, concurrency and metric
  production; verify with `cargo test -p sciencekit_decomposition`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on covariance merging and streaming finalization); verify
  with `mdbook build docs`.
- [ ] 4.4 Record the GPU/memmap/export non-goals in `docs/PRD.md`.
- [ ] 4.5 Independent review by a separate agent validating every scenario in
  `incremental-pca`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
