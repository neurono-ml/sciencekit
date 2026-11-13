# Proposal

## Why

The PRD requires every estimator and pipeline to export to ONNX and to load ONNX models
trained elsewhere. The library has no ONNX graph builder, no opset mapping and no per-model
export logic. Export is the first half: turn a fitted model's state/configuration into a
valid ONNX graph.

## What Changes

- New `sciencekit_onnx` crate with the `SKToOnnx<F>` trait and an `SKOnnxModel` byte
  container.
- A graph builder over the ONNX protobuf model: opset version, inputs/outputs, initializers,
  nodes and metadata.
- Per-family exporters: preprocessing scalers/encoders/imputers, linear/logistic models,
  linear and kernel SVMs, trees and tree ensembles (ONNX-ML), neighbours, clustering,
  decomposition, isolation forest and pipelines.
- Dtype mapping (`f32`/`f64`), feature-count and class-table metadata.
- A dependency spike to choose the protobuf encoding (`prost` + vendored ONNX proto,
  `onnx-protobuf`, or a maintained helper crate) under Apache-2.0 and MSRV 1.85.
- Export validation by loading the produced graph with a compatible ONNX runtime and
  comparing predictions on fixtures.
- Explicit non-goals: ONNX import (separate change), native-type import, training, GPU and
  Python.
- Companion tests: graph validity, per-family prediction parity, dtype coverage, metadata,
  unsupported-family errors and concurrent export.

## Capabilities

### New Capabilities

- `onnx-export`: the ONNX graph builder, the `SKToOnnx` contract, per-family operator
  mapping, dtype/metadata rules and export validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_onnx` workspace member; depends on the model crates and
  `sciencekit_interop`.
- **APIs:** additive.
- **Dependencies:** protobuf encoding (spike), optional runtime for validation
  (`tract-onnx 0.22`, MSRV 1.85; `tract-onnx 0.23` requires Rust 1.91).
- **Downstream:** unblocks the ONNX import path and production interoperability.
- **Acceptance criteria:** exported graphs load and predict, per-family parity, large/small
  models and concurrent export.
