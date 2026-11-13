# Proposal

## Why

Importing ONNX or Safetensors currently yields a generic predictor or an export state, not a
native model that can be inspected, retrained or composed in a pipeline. The PRD records
native import and resumed training as the next interoperability step. This change converts a
documented subset of imported models into native types and enables continued fitting where
the algorithm supports it.

## What Changes

- Add a native conversion path from Safetensors/ONNX into native estimator/model types for a
  documented subset of families (linear/logistic, scalers/encoders, trees/forests where the
  state maps exactly).
- Reuse `SKImportable` for Safetensors states; add an ONNX→native converter for supported
  operator graphs.
- Enable warm-start/continued fitting for models whose native configuration can be
  reconstructed (for example linear/logistic `partial_fit` or a new `fit` with initial
  state).
- Preserve class tables, feature counts, backend/solver and hyperparameters.
- Validation: unsupported graph/family, ambiguous weights, shape mismatch and missing
  metadata; conversions never produce a partially initialized model.
- Explicit non-goals: arbitrary ONNX graphs, neural networks, LoRA (separate change),
  GPU training and changing the generic predictor path.
- Companion tests: exported-library round-trip into native, supported ONNX fixtures,
  resumed-training equivalence and unsupported-graph errors.

## Capabilities

### New Capabilities

- `native-model-import`: ONNX/Safetensors → native type conversion and resumed-training
  support for the documented subset.

### Modified Capabilities

None.

## Impact

- **Code:** interop/ONNX conversion modules plus native constructors in the model crates.
- **APIs:** additive; the generic ONNX predictor remains.
- **Dependencies:** none new beyond Wave 7 interop/ONNX.
- **Downstream:** enables retraining workflows, the CLI and LoRA adapters.
- **Acceptance criteria:** round-trip native conversion, resumed-training equivalence,
  unsupported graphs explicit and concurrent conversion deterministic.
