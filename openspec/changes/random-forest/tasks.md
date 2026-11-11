# Tasks

## 1. Crate and shared bootstrap

- [x] 1.1 Create `crates/sciencekit_ensemble` with `Cargo.toml`, `lib.rs`, the
  `bootstrap` module and the `random_forest` dispatcher; add it to
  `[workspace.members]`; verify `cargo test -p sciencekit_ensemble` compiles.
- [x] 1.2 Write and implement the seeded bootstrap sampler (row indices with replacement,
  seeded SplitMix64, deterministic per-tree seeds) and the OOB mask, with tests for
  reproducibility, sample size, replacement and the no-bootstrap case; verify with
  `cargo test -p sciencekit_ensemble bootstrap`.

## 2. Forest configuration and fitting

- [x] 2.1 Implement `SKRandomForestClassifierBuilder` and `SKRandomForestRegressorBuilder`
  with the documented defaults and invalid-value tests (`number_of_trees` zero, invalid
  sample fraction, OOB without bootstrap); verify with
  `cargo test -p sciencekit_ensemble forest_builder`.
- [x] 2.2 Implement parallel per-tree fitting over bootstrap samples with deterministic
  seeds and lock-free slots; verify parallel-equals-sequential and repeated-fit
  determinism; verify with `cargo test -p sciencekit_ensemble forest_fit`.
- [x] 2.3 Implement classifier probability averaging and label prediction with canonical
  tie-breaking; verify a hand-computed forest; verify with
  `cargo test -p sciencekit_ensemble forest_classifier`.
- [x] 2.4 Implement regressor response averaging; verify a hand-computed forest; verify
  with `cargo test -p sciencekit_ensemble forest_regressor`.

## 3. OOB, importances and representation

- [x] 3.1 Implement OOB scoring (accuracy / R²) over rows excluded from each tree's
  bootstrap, with tests for a fully covered OOB row, no scored rows and the disabled
  case; verify with `cargo test -p sciencekit_ensemble forest_oob`.
- [x] 3.2 Implement mean forest feature importances with normalization and a test on a
  separating feature; verify with `cargo test -p sciencekit_ensemble forest_importances`.
- [x] 3.3 Add dense-vs-sparse fit/predict equivalence tests and a test proving sparse
  fitting does not densify; verify with
  `cargo test -p sciencekit_ensemble forest_representation`.

## 4. Validation, determinism and acceptance

- [x] 4.1 Add tests for empty input, feature-count mismatch, non-finite values and
  single-class classification, asserting the exact central error variant; verify with
  `cargo test -p sciencekit_ensemble forest_validation`.
- [x] 4.2 Add a `Send + Sync` assertion and a concurrent-prediction test sharing one
  forest; verify with `cargo test -p sciencekit_ensemble forest_concurrency`.
- [x] 4.3 Add `acceptance_tests/` covering tiny and large data (at least 20 000 rows),
  classifier and regressor, OOB on/off, dense/sparse, thread-count invariance and metric
  production through `sciencekit_metrics`; verify with
  `cargo test -p sciencekit_ensemble`.

## 5. Documentation, changelog and review

- [ ] 5.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the ensemble crate, both forests and OOB scoring.
- [ ] 5.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the random-forest chapter, register it in `SUMMARY.md`, add
  a `Development Explained` chapter covering bagging, feature subsampling, OOB and
  parallel aggregation, update `maintaining.md`, and verify with `mdbook build docs`.
- [ ] 5.3 Record the deferred missing-value routing, streaming, `memmap2` and export work
  in `docs/PRD.md`; verify the amended paragraphs render.
- [ ] 5.4 Independent review by a separate agent validating every scenario in
  `random-forests` against the code and tests; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
