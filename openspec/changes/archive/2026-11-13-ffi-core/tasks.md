# Tasks

## 1. ABI core

- [x] 1.1 Create `crates/sciencekit_ffi` with `cdylib`/`staticlib`, status/version/error
  modules and a null-handle test; verify `cargo test -p sciencekit_ffi status`.
- [x] 1.2 Implement the handle table and create/free for builders/models; verify with
  `cargo test -p sciencekit_ffi handle`.
- [x] 1.3 Implement `SKFfiArray` validation and borrowed input descriptors; verify with
  `cargo test -p sciencekit_ffi array`.

## 2. Operations

- [x] 2.1 Implement builder setters and fit for the supported families; verify with
  `cargo test -p sciencekit_ffi fit`.
- [x] 2.2 Implement predict/transform/score and owned output arrays with free; verify with
  `cargo test -p sciencekit_ffi predict`.
- [x] 2.3 Implement model save/load through the interop formats; verify with
  `cargo test -p sciencekit_ffi persistence`.

## 3. Safety and packaging

- [x] 3.1 Add `catch_unwind` to every entry point and a panic-to-status test; verify with
  `cargo test -p sciencekit_ffi panic`.
- [x] 3.2 Generate/check `sciencekit.h` and add a C smoke test that fits and predicts;
  verify with the C build test.
- [x] 3.3 Add concurrent predict, invalid-array and version-query tests; verify with
  `cargo test -p sciencekit_ffi concurrency`.
- [x] 3.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 3.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on the C ABI and memory ownership); verify with
  `mdbook build docs`.
- [x] 3.6 Independent review by a separate agent validating every scenario in `ffi-abi`;
  address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
