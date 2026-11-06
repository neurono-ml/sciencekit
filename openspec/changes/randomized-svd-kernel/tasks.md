# Tasks

## 1. Module and result

- [ ] 1.1 Create `crates/sciencekit_math/src/decomposition/` with dispatcher `mod.rs`, the
  `SKRandomizedSvdResult` container and re-exports; verify
  `cargo test -p sciencekit_math randomized` compiles.
- [ ] 1.2 Write and implement the seeded SplitMix64 projection generator (Rademacher or
  Gaussian), with reproducibility and shape tests; verify with
  `cargo test -p sciencekit_math randomized_projection`.

## 2. Dense path

- [ ] 2.1 Write and implement the dense range finder (`XΩ`, QR, power iterations), with a
  low-rank reconstruction test and a power-iteration improvement test; verify with
  `cargo test -p sciencekit_math randomized_dense`.
- [ ] 2.2 Implement the final small SVD (`B = QᵀX`, lift `U`, truncate) and the deterministic
  sign convention, with shape, singular-value ordering and sign tests; verify with
  `cargo test -p sciencekit_math randomized_dense`.

## 3. Sparse path

- [ ] 3.1 Implement the sparse `XΩ` and `XᵀY` products over CSR without densifying, with a
  sparse-vs-dense equivalence test and a wide-sparse no-densification test; verify with
  `cargo test -p sciencekit_math randomized_sparse`.
- [ ] 3.2 Add a large sparse fixture proving the dense intermediate is limited to
  `m × (k + oversampling)`; verify with `cargo test -p sciencekit_math randomized_sparse`.
- [ ] 3.3 Implement the optional centering vector with the implicit centered products,
  and test centered dense vs explicit centering, centered sparse vs centered dense and
  no densification; verify with
  `cargo test -p sciencekit_math randomized_centered`.

## 4. Validation, determinism and acceptance

- [ ] 4.1 Add tests for empty input, zero/excessive components, zero power iterations and
  invalid oversampling, asserting the exact error variant; verify with
  `cargo test -p sciencekit_math randomized_validation`.
- [ ] 4.2 Add repeated-call determinism, concurrent-call determinism and a `Send + Sync`
  assertion; verify with `cargo test -p sciencekit_math randomized_concurrency`.
- [ ] 4.3 Add acceptance tests for tiny and large dense/sparse matrices, decaying spectra,
  every component count boundary and the documented tolerance; verify with
  `cargo test -p sciencekit_math`.
- [ ] 4.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.

## 5. Documentation and review

- [ ] 5.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the randomized SVD kernel.
- [ ] 5.2 Add a `Development Explained` subsection to the decomposition chapter covering
  randomized range finding, power iterations, sparse products and the sign convention;
  verify with `mdbook build docs`.
- [ ] 5.3 Independent review by a separate agent validating every scenario in
  `randomized-svd` against the code and tests; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
