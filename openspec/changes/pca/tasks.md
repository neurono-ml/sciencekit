# Tasks

## 1. Crate and centering

- [x] 1.1 Create `crates/sciencekit_decomposition` with `Cargo.toml`, `lib.rs` and the
  `pca` dispatcher; add it to `[workspace.members]`; verify
  `cargo test -p sciencekit_decomposition` compiles.
- [x] 1.2 Implement centering and mean storage with a hand-computed zero-mean test; verify
  with `cargo test -p sciencekit_decomposition pca_centering`.

## 2. Solvers

- [x] 2.1 Implement the full-SVD solver with components, singular values, explained
  variance and ratio, and a reconstruction test; verify with
  `cargo test -p sciencekit_decomposition pca_full`.
- [x] 2.2 Implement the randomized solver using `sk_randomized_svd`, with full-vs-randomized
  agreement and reproducibility tests; verify with
  `cargo test -p sciencekit_decomposition pca_randomized`.
- [x] 2.3 Implement the covariance-eigh solver with sorting and full-SVD agreement tests;
  verify with `cargo test -p sciencekit_decomposition pca_covariance`.
- [x] 2.4 Implement the `Auto` resolution and expose the resolved solver, with the small/
  large/covariance branches tested; verify with
  `cargo test -p sciencekit_decomposition pca_solver`.

## 3. Estimator and transforms

- [x] 3.1 Implement `SKPCABuilder` with the documented defaults and invalid-value tests;
  verify with `cargo test -p sciencekit_decomposition pca_builder`.
- [x] 3.2 Implement `SKUnsupervisedFit` and the model accessors (components, variance,
  ratio, singular values, mean); verify with
  `cargo test -p sciencekit_decomposition pca_fit`.
- [x] 3.3 Implement `transform`, `inverse_transform` and `fit_transform`, including the
  whitening scaling and the full-component reconstruction test; verify with
  `cargo test -p sciencekit_decomposition pca_transform`.
- [ ] 3.4 Add the sparse randomized test (no dense `Xc`, equals dense centered), the
  `Full`/`CovarianceEigh` sparse rejection with alternatives, the explicit
  `allow_densification(true)` warning path and the feature-count mismatch test; verify
  with `cargo test -p sciencekit_decomposition pca_validation`.
- [ ] 3.5 Add repeated-fit determinism, concurrent transform and `Send + Sync` tests;
  verify with `cargo test -p sciencekit_decomposition pca_concurrency`.

## 4. Acceptance and review

- [ ] 4.1 Add `acceptance_tests/` covering tiny and large dense data, all solvers, every
  component count boundary, whitening, determinism, concurrency and metric production;
  verify with `cargo test -p sciencekit_decomposition`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on centering, SVD, eigh, whitening and the dense-only
  exception); verify with `mdbook build docs`.
- [ ] 4.4 Record the sparse-full/eigh solver, kernel PCA, incremental PCA, GPU, `memmap2`
  and export non-goals in `docs/PRD.md`.
- [ ] 4.5 Independent review by a separate agent validating every scenario in
  `principal-component-analysis`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
