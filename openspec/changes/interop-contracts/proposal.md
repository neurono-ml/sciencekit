# Proposal

## Why

Every Wave 1–6 model keeps its learned state in private Rust fields, and no central contract
describes that state for export or import. Without one, Safetensors, JSON debug and ONNX
would each invent a different introspection mechanism. A format-agnostic model state —
metadata plus named tensors — is the smallest foundation that lets every format share one
export path.

## What Changes

- Add `SKExportable<F>` and `SKImportable<F>` to `sciencekit_common`.
- Add `SKModelState<F>`: estimator metadata, a named tensor map and typed hyperparameters.
- Add `SKTensor<F>`: a stable name, a dtype, a shape and the values.
- Add `SKModelMetadata`: estimator name, format version, scalar dtype, feature count,
  class labels and a hyperparameter map.
- Add `SKParameterValue` (bool/integer/float/string/list) so metadata stays serializable
  without making `sciencekit_common` depend on serde.
- Add `SKTensorDType` (`F32`, `F64`, `I64`, `Bool`) and shape/dtype validation.
- No concrete file format, no ONNX and no Python code in this change.

## Capabilities

### New Capabilities

- `model-export-contracts`: the export/import trait pair, the model state container, the
  named tensor representation, metadata and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `sciencekit_common` gains an `interop` module and re-exports; a new
  `crates/sciencekit_interop` skeleton is created for the format adapters.
- **APIs:** additive. Every fitted model will implement `SKExportable` in
  `model-export-retrofit`.
- **Dependencies:** none new.
- **Downstream:** unblocks Safetensors, JSON debug, ONNX export/import and Python
  introspection.
- **Acceptance criteria:** contract tests with a mock model, shape/dtype/name validation,
  state round-trip through the in-memory representation and concurrent read-only access.
