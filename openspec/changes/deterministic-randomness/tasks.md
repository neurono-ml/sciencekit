# Tasks

## 1. Shared generator

- [x] 1.1 Create `sciencekit_common/src/random/` with dispatcher `mod.rs`, the SplitMix64
  generator moved from `sciencekit_linear_model`, and golden-vector tests for `next_u64`;
  verify with `cargo test -p sciencekit_common random`.
- [x] 1.2 Add `next_f64`, `uniform`, `normal` (Box-Muller with cached spare), `shuffle`
  and `sample_indices_without_replacement`, with range, permutation and moment tests; verify
  with `cargo test -p sciencekit_common random`.
- [x] 1.3 Add `sk_derive_seed` and domain constants, with distinctness and
  order-independence tests; verify with `cargo test -p sciencekit_common sub_seed`.
- [x] 1.4 Re-export the generator and derive helper from `sciencekit_common`; verify with
  `cargo test -p sciencekit_common`.

## 2. Linear-model refactor

- [x] 2.1 Replace the private SGD generator with the shared one and delete the duplicate
  implementation; verify the existing seeded SGD tests with
  `cargo test -p sciencekit_linear_model sgd`.
- [x] 2.2 Add a cross-crate test proving the same seed produces the same shuffle order in
  the shared generator and in SGD; verify with `cargo test -p sciencekit_linear_model`.
- [x] 2.3 Add `random_seed(u64)` (default `0`) to `SKLogisticRegression`, `SKLasso` and
  `SKElasticNet`, replace the private xorshift64* permutations (SAGA and coordinate-descent
  random selection) with `SKRandomGenerator` sub-seeds, and test the default, explicit and
  distinct-seed behaviour; verify with `cargo test -p sciencekit_linear_model saga` and
  `cargo test -p sciencekit_linear_model coordinate`.
- [x] 2.4 Prove stable collection order: fit the same estimator under one and four rayon
  threads and assert identical coefficient rows by index; verify with
  `cargo test -p sciencekit_linear_model`.

## 3. Documentation and review

- [x] 3.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the shared generator and the `random_seed` rule.
- [x] 3.2 Add a `Development Explained` subsection covering SplitMix64, sub-seeds,
  thread-count invariance and the reproducibility scope; verify with `mdbook build docs`.
- [x] 3.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.4 Independent review by a separate agent validating every scenario in
  `deterministic-randomness`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
