# Tasks

## 1. Dependency spike and crate

- [x] 1.1 Spike the protobuf backend (`prost` + vendored proto, `onnx-protobuf`,
  `axonml-onnx`) under MSRV 1.85 and Apache-2.0-compatible licensing; record the choice and
  evidence in `design.md`.
- [x] 1.2 Create `crates/sciencekit_onnx` with `Cargo.toml`, `lib.rs` and the graph
  dispatcher; add it to `[workspace.members]`; verify `cargo test -p sciencekit_onnx`
  compiles.
- [x] 1.3 Implement the graph builder (inputs, initializers, nodes, outputs, metadata) and
  a hand-checkable identity graph; verify with `cargo test -p sciencekit_onnx graph`.

## 2. Core exporters

- [x] 2.1 Implement preprocessing exporters (scalers, encoders, imputers) with graph
  validity and prediction-parity tests; verify with
  `cargo test -p sciencekit_onnx export_preprocessing`.
- [x] 2.2 Implement linear/logistic exporters with MatMul/Add/link operators and f32/f64
  parity tests; verify with `cargo test -p sciencekit_onnx export_linear`.
- [x] 2.3 Implement tree/forest/boosting exporters via ONNX-ML tree ensembles and parity
  tests; verify with `cargo test -p sciencekit_onnx export_tree`.

## 3. Remaining families

- [ ] 3.1 Implement SVM exporters (linear/RBF) and parity tests; verify with
  `cargo test -p sciencekit_onnx export_svm`.
- [x] 3.2 Implement neighbour, clustering, decomposition and outlier exporters with parity
  tests; verify with `cargo test -p sciencekit_onnx export_remaining`.
- [ ] 3.3 Implement pipeline composition and grid-search best-model export; verify with
  `cargo test -p sciencekit_onnx export_pipeline`.
- [ ] 3.4 Record the family/operator coverage matrix and add unsupported-family errors;
  verify with `cargo test -p sciencekit_onnx export_coverage`.

## 4. Validation, acceptance and review

- [ ] 4.1 Load exported graphs with `tract-onnx 0.22` and compare predictions on fixtures;
  verify with `cargo test -p sciencekit_onnx onnx_runtime`.
- [ ] 4.2 Add metadata, dtype, large/small model and concurrent-export tests; verify with
  `cargo test -p sciencekit_onnx`.
- [ ] 4.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.4 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on ONNX graphs, opsets and operator mapping); verify with
  `mdbook build docs`.
- [ ] 4.5 Independent review by a separate agent validating every scenario in
  `onnx-export`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
