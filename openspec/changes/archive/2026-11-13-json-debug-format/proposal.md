# Proposal

## Why

Binary Safetensors is the production format, but debugging a model requires reading its
parameters and tensors as text. The PRD asks for a human-readable JSON format using the
same central export state, so users can inspect a fitted model without a binary tool.

## What Changes

- Add a JSON debug writer/reader to `sciencekit_interop` over `SKModelState`.
- Human-readable schema: metadata, typed hyperparameters and named tensors with shape and
  flat values.
- Pretty and compact output modes; stable key ordering from the state maps.
- Import back into `SKModelState` for round-trip tests.
- Validation: JSON schema/version, tensor shape/value length, dtype and duplicate names.
- Explicit non-goals: binary Safetensors, ONNX and Python.
- Companion tests: hand-inspectable output, round-trip, large/small tensors, compact/pretty
  equivalence, corrupt JSON errors and concurrency.

## Capabilities

### New Capabilities

- `json-debug-format`: the human-readable model-state JSON schema, writer, reader and
  validation.

### Modified Capabilities

None.

## Impact

- **Code:** `sciencekit_interop` gains a `json` module.
- **APIs:** additive.
- **Dependencies:** `serde_json` (already planned for the interop crate).
- **Downstream:** gives users a debuggable model representation and a fixture format for
  tests.
- **Acceptance criteria:** round-trip fidelity, readable fixtures, large tensor handling
  and corruption errors.
