# Tasks

## 1. Native import core

- [x] 1.1 Define `SKNativeImport<F>` and the conversion result/error surface; verify with
  `cargo test -p sciencekit_interop native_import`.
- [x] 1.2 Wire Safetensors/JSON state import through `SKImportable` for linear/preprocessing
  families and test round-trip predictions; verify with
  `cargo test -p sciencekit_interop native_safetensors`.

## 2. ONNX conversion

- [x] 2.1 Implement the ONNX native importer for the documented operator subset and
  initializer mapping; verify with `cargo test -p sciencekit_onnx native_import`.
- [x] 2.2 Add preprocessing/linear/tree fixtures and unsupported-operator errors; verify
  with `cargo test -p sciencekit_onnx native_coverage`.

## 3. Retraining

- [x] 3.1 Implement warm-start/continued fitting for supported linear/logistic models;
  verify with `cargo test -p sciencekit_linear_model native_retrain`.
- [x] 3.2 Add explicit `UnsupportedOperation` for non-retrainable families; verify with
  `cargo test -p sciencekit_interop native_retrain`.

## 4. Acceptance and review

- [x] 4.1 Add concurrent conversion, shape/class-table validation and no-partial-model
  tests; verify with `cargo test -p sciencekit_interop native_validation`.
- [x] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on native import and retraining); verify with
  `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `native-model-import`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
