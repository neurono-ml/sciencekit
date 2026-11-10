# Tasks

## 1. State and tensors

- [x] 1.1 Create `sciencekit_common/src/interop/` with dispatcher `mod.rs`,
  `SKTensorDType`, `SKTensor` and shape/dtype validation; verify with
  `cargo test -p sciencekit_common interop_tensor`.
- [x] 1.2 Implement `SKParameterValue`, `SKModelMetadata` and `SKModelState` with unique-
  name and metadata validation; verify with `cargo test -p sciencekit_common interop_state`.

## 2. Traits and round-trip

- [x] 2.1 Add `SKExportable`/`SKImportable`, a mock model and an export/import round-trip
  test; verify with `cargo test -p sciencekit_common interop_traits`.
- [x] 2.2 Add wrong-estimator, wrong-version, wrong-feature-count and unsupported-model
  tests; verify with `cargo test -p sciencekit_common interop_validation`.
- [x] 2.3 Add a concurrent-export test and `Send + Sync` assertions; verify with
  `cargo test -p sciencekit_common interop_concurrency`.

## 3. Documentation and review

- [x] 3.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the export/import contracts.
- [x] 3.2 Add a `Development Explained` subsection on the model state, stable tensor names
  and the no-serde decision; verify with `mdbook build docs`.
- [x] 3.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.4 Independent review by a separate agent validating every scenario in
  `model-export-contracts`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
