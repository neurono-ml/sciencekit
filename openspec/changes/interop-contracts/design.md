# Design

## Context

See `proposal.md` — Why. `sciencekit_common` is the dependency-free vocabulary crate; it
already owns the scalar, data, label and error contracts. The export contract must live
here so every model crate can implement it without depending on a format crate (which would
create a cycle with `sciencekit_interop`).

## Goals / Non-Goals

**Goals:**
- One format-agnostic state for Safetensors, JSON and Python introspection.
- No serde dependency in common.
- Stable tensor names and dtype/shape validation.

**Non-Goals:**
- File formats, compression, ONNX graphs or Python bindings.
- Automatic export for every model (that is `model-export-retrofit`).
- Hyperparameter schemas beyond a typed value map.

## Decisions

### Decision 1 — State = metadata + named tensors + parameters

```rust
pub struct SKModelState<F: SKFloat> {
    pub metadata: SKModelMetadata,
    pub tensors: BTreeMap<String, SKTensor<F>>,
    pub parameters: BTreeMap<String, SKParameterValue>,
}
```

`BTreeMap` gives deterministic ordering for reproducible files and JSON. Tensors are keyed
by a stable dotted path (`linear.coefficients`, `tree.nodes.feature`, ...), documented per
model in the retrofit change.

### Decision 2 — Tensor carries dtype, shape and values

`SKTensor<F>` stores `Vec<F>` plus an `SKTensorDType` (`F32`, `F64`, `I64`, `Bool`) and a
shape. Floating tensors use `F`; integer/bool tensors for class tables and tree indices are
represented without forcing them into `F` (the values are stored in an enum or a separate
vector). The exact representation is an implementation detail; the contract is name, dtype,
shape and count.

### Decision 3 — Metadata without serde

`SKModelMetadata` holds strings, an integer version, the scalar dtype, feature count,
optional class labels and a `BTreeMap<String, SKParameterValue>`. `SKParameterValue` is a
small enum (`Bool`, `Integer`, `Float`, `String`, `List`) so format adapters can serialize
it with their own writer and common stays serde-free.

### Decision 4 — Export and import traits

```rust
pub trait SKExportable<F: SKFloat> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError>;
}

pub trait SKImportable<F: SKFloat>: Sized {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError>;
}
```

Export is model-side only. Import validates the estimator name, version and feature count
before touching tensors. A model that cannot be reconstructed returns
`UnsupportedOperation` rather than a half-built value.

### Decision 5 — Validation

Construction validates unique tensor names, shape/data length, dtype/scalar agreement,
feature count and class-label uniqueness. Errors use the central taxonomy
(`InvalidHyperparameter`, `ShapeMismatch`, `UnsupportedRepresentation`).

### Decision 6 — File layout

```
crates/sciencekit_common/src/interop/
  mod.rs, state.rs, tensor.rs, metadata.rs, parameter.rs, traits.rs, *_tests.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **A generic tensor type can become a mini-serialization framework** → keep it small:
  metadata, tensors and typed parameters only.
- **Stable names are a compatibility promise** → document them in the retrofit change and
  test round-trips.
- **Import is optional per model** → `UnsupportedOperation` is explicit.
- **No serde in common** → format crates own their serialization; common remains light.

## Migration Plan

Additive: new module and traits. No existing API changes.

## Open Questions

None blocking. A schema registry and automatic derives are future conveniences.
