# Design

## Context

See `proposal.md` — Why. `interop-contracts` and `model-export-retrofit` define/implement
`SKImportable` for Safetensors states. `onnx-import` provides a generic predictor. This
change adds native conversion on top of both, without changing the generic path.

## Goals / Non-Goals

**Goals:**
- Native models from supported states/graphs.
- Continued training where the algorithm allows it.
- Explicit unsupported cases.

**Non-Goals:**
- Arbitrary ONNX, neural networks, LoRA (separate), GPU training.

## Decisions

### Decision 1 — Two conversion paths

- Safetensors/JSON state: use `SKImportable` where implemented by the model crate.
- ONNX: a `SKOnnxNativeImporter` maps a documented operator subset/initializer names to a
  native configuration and state. It reuses the ONNX parser from `onnx-import`.

Both return `SKNativeImport<F>` (`SKNativeImport::Classifier`, `Regressor`, `Transformer`
or `Clusterer`) so the caller can inspect the family before using it.

### Decision 2 — Supported subset

Initial coverage: preprocessing scalers/encoders/imputers, linear/logistic/ridge/lasso/
elastic-net, and tree/forest ensembles with ONNX-ML attributes. Other families return
`UnsupportedOperation`; the matrix is documented and grows in later changes.

### Decision 3 — Continued training

Linear/logistic models support warm start (`fit` with initial coefficients) or `partial_fit`
where the algorithm supports it. Tree/ensemble models are inference-only for now. The
native importer records whether retraining is supported.

### Decision 4 — Validation

Validate estimator name/version, feature count, class table, tensor names/shapes/dtypes and
ONNX operator subset before constructing. A failed conversion leaves no model.

### Decision 5 — File layout

```
crates/sciencekit_interop/src/native/
  mod.rs, safetensors.rs, onnx.rs, coverage.rs, error.rs, *_tests.rs
crates/sciencekit_onnx/src/native/
  mod.rs, importer.rs, ops.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **ONNX graph variety** → a documented subset and explicit errors.
- **State/name compatibility** → reuse the retrofit names and round-trip tests.
- **Retraining semantics** → warm start documented per family; no silent reset.
- **Two import paths** → one `SKNativeImport` result type.

## Migration Plan

Additive: new modules and constructors. No existing API changes.

## Open Questions

None blocking. Neural-network/LoRA support is the `lora-adapters` change.
