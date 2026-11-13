# Proposal

## Why

The PRD requires imported ONNX models to load as a generic predictor, usable standalone or
inside a pipeline, without converting them into native library types. Export alone is only
half of interoperability; import lets users run models trained elsewhere and makes exported
graphs round-trippable through an external runtime.

## What Changes

- Add an ONNX import path to `sciencekit_onnx`: parse an ONNX model and run it as a
  generic predictor.
- A `SKOnnxPredictor<F>` that implements the regressor and/or classifier prediction
  surfaces based on the model's output shape and metadata.
- Input/output mapping: feature matrix to ONNX input tensor, output tensor(s) to responses
  or class probabilities/labels.
- A supported-operator contract with explicit `UnsupportedOperation` for operator/model
  constructs outside the supported set.
- A dependency spike: prefer `tract-onnx 0.22` (MSRV 1.85) if its transitive MSRV holds;
  otherwise implement a minimal interpreter over the parsed protobuf for a documented op
  subset.
- Validation: missing inputs, shape/dtype mismatch, unsupported ops, multiple outputs and
  batch-size handling.
- Explicit non-goals: native-type import, resumed training, LoRA, GPU and Python.
- Companion tests: import exported library models, external ONNX fixtures, classifier and
  regressor outputs, unsupported-op errors, shape validation and concurrency.

## Capabilities

### New Capabilities

- `onnx-import`: ONNX model loading, the generic predictor contract, input/output mapping,
  supported operators and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_onnx` gains the import module and predictor.
- **APIs:** additive.
- **Dependencies:** `tract-onnx 0.22` (MSRV 1.85) or a minimal interpreter; feature-gated.
- **Downstream:** completes ONNX interoperability and gives pipelines a foreign-model
  predictor.
- **Acceptance criteria:** exported and external fixtures predict, unsupported ops are
  explicit, shapes/dtypes validate and concurrent prediction is deterministic.
