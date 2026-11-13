# Tasks

## 1. Runtime spike and model wrapper

- [x] 1.1 Spike `tract-onnx 0.22` on MSRV 1.85 with a tiny ONNX fixture; record whether
  it builds and runs, otherwise choose the minimal interpreter; update `design.md`.
- [x] 1.2 Implement `SKOnnxModel` parsing/metadata (input/output names, shapes, dtypes) and
  invalid-bytes tests; verify with `cargo test -p sciencekit_onnx onnx_model`.
- [x] 1.3 Implement the runtime/interpreter adapter and an identity-graph inference test;
  verify with `cargo test -p sciencekit_onnx onnx_runtime`.

## 2. Predictor

- [x] 2.1 Implement the regressor predictor and shape/dtype validation; verify with
  `cargo test -p sciencekit_onnx onnx_regressor`.
- [x] 2.2 Implement the classifier probabilities/labels path and ambiguity rejection;
  verify with `cargo test -p sciencekit_onnx onnx_classifier`.
- [x] 2.3 Add batch of one, large batch and row-by-row agreement tests; verify with
  `cargo test -p sciencekit_onnx onnx_batch`.

## 3. Operator coverage and validation

- [x] 3.1 Implement the documented supported-operator interpreter path (or configure tract)
  and unsupported-op errors; verify with `cargo test -p sciencekit_onnx onnx_operators`.
- [ ] 3.2 Add external fixture tests, missing-input, multiple-output and non-finite-input
  tests; verify with `cargo test -p sciencekit_onnx onnx_validation`.

## 4. Acceptance and review

- [ ] 4.1 Import the exported library models (linear, scaler, tree, pipeline) and compare
  predictions; verify with `cargo test -p sciencekit_onnx onnx_round_trip`.
- [ ] 4.2 Add concurrent-prediction and `Send + Sync` tests; verify with
  `cargo test -p sciencekit_onnx onnx_concurrency`.
- [ ] 4.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.4 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on generic ONNX inference and supported ops); verify with
  `mdbook build docs`.
- [ ] 4.5 Independent review by a separate agent validating every scenario in
  `onnx-import`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
