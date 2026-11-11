# Tasks

## 1. Module and fit

- [x] 1.1 Create `crates/sciencekit_decomposition/src/truncated_svd/` with dispatcher
  `mod.rs`, config and model; verify `cargo test -p sciencekit_decomposition truncated`
  compiles.
- [x] 1.2 Implement the randomized-SVD fit and model state (components, singular values,
  explained variance and ratio), with a rank-1/rank-2 reconstruction test; verify with
  `cargo test -p sciencekit_decomposition truncated_fit`.
- [x] 1.3 Implement dense/sparse parity and a no-densification test, including the
  documented tolerance; verify with
  `cargo test -p sciencekit_decomposition truncated_representation`.

## 2. Transforms and estimator

- [x] 2.1 Implement `transform` and `inverse_transform` for dense and sparse views, with
  shape and reconstruction tests; verify with
  `cargo test -p sciencekit_decomposition truncated_transform`.
- [x] 2.2 Implement the builder with the documented defaults and invalid-value tests;
  verify with `cargo test -p sciencekit_decomposition truncated_builder`.
- [x] 2.3 Implement `SKUnsupervisedFit` and the model accessors; verify with
  `cargo test -p sciencekit_decomposition truncated_fit`.
- [ ] 2.4 Add validation tests for empty input, zero/excessive components, invalid
  iterations/tolerance/algorithm and feature-count mismatch; verify with
  `cargo test -p sciencekit_decomposition truncated_validation`.
- [ ] 2.5 Add repeated-fit determinism, concurrent transform and `Send + Sync` tests;
  verify with `cargo test -p sciencekit_decomposition truncated_concurrency`.

## 3. Acceptance and review

- [ ] 3.1 Add `acceptance_tests/` covering tiny and large dense/sparse matrices, every
  component boundary, determinism, concurrency and metric production; verify with
  `cargo test -p sciencekit_decomposition`.
- [ ] 3.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on uncentered low-rank factorization and the sparse
  path); verify with `mdbook build docs`.
- [ ] 3.4 Record the ARPACK/GPU/memmap/export non-goals in `docs/PRD.md`.
- [ ] 3.5 Independent review by a separate agent validating every scenario in
  `truncated-svd`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
