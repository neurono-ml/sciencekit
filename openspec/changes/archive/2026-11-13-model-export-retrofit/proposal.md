# Proposal

## Why

`interop-contracts` defines how a fitted model exposes its state, but no model implements
it. Until every Wave 1–6 model exports a stable state, Safetensors, JSON debug, ONNX and the
Python bindings have nothing to read. This change retrofits the contract across the
library's fitted models and pipelines.

## What Changes

- Implement `SKExportable<F>` for every existing fitted model and pipeline model.
- Implement `SKImportable<F>` where the state is sufficient to reconstruct the model
  exactly; models that cannot be reconstructed return `UnsupportedOperation`.
- Define stable tensor names and metadata per estimator family:
  preprocessing scalers/encoders, impute, linear models, neighbours, SVM, trees,
  ensembles, clustering, decomposition, outlier detection, pipelines and grid-search
  winners.
- Preserve hyperparameters, feature counts, class/label tables, resolved backend, solver
  and other configuration needed for a faithful round-trip.
- Add per-family round-trip tests: export → import → predictions equal within tolerance;
  stable tensor names; metadata completeness.
- Add a shared export-test helper (in a dev-only module) so every family follows the same
  checks.
- No file format, ONNX graph or Python code in this change.

## Capabilities

### New Capabilities

- `model-export-coverage`: the export/import implementation for every fitted model
  family, its stable tensor names, metadata and round-trip guarantees.

### Modified Capabilities

None.

## Impact

- **Code:** every algorithm crate gains `export`/`import` modules on its model types; a
  dev-only test helper is shared.
- **APIs:** additive trait implementations.
- **Dependencies:** none new; `sciencekit_common` only.
- **Downstream:** unblocks Safetensors, JSON debug, ONNX and Python bindings.
- **Acceptance criteria:** every fitted model exports a non-empty state, imports where
  supported, and round-trips predictions on small and large data; concurrent export is
  deterministic.
