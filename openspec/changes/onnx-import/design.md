# Design

## Context

See `proposal.md` — Why. `sciencekit_onnx` already owns export and the protobuf backend.
Import needs a runtime/interpreter. MSRV findings: `tract-onnx 0.22` declares Rust 1.85;
`0.23` requires 1.91; `ort` requires 1.92. Tract is the preferred option if its transitive
dependencies stay on 1.85; otherwise the minimal interpreter is the fallback.

## Goals / Non-Goals

**Goals:**
- Load exported and external ONNX models reliably.
- A generic predictor with clear classifier/regressor adaptation.
- Explicit unsupported-operator behavior.

**Non-Goals:**
- Native-type import, training/LoRA, GPU and Python.
- Every ONNX operator and dynamic shapes.
- Quantized models.

## Decisions

### Decision 1 — Runtime choice

A spike evaluates `tract-onnx 0.22` on MSRV 1.85 and on the exported fixtures. If it holds,
use it behind an `onnx-runtime` feature. If not, implement a minimal interpreter for a
documented op subset (MatMul, Add, Sub, Mul, Div, Relu, Sigmoid, Softmax, Reshape, Flatten,
Gather, Concat, Cast, Constant, ReduceMean, ArgMax, TopK). The public API is the same either
way.

### Decision 2 — Model wrapper and predictor

`SKOnnxModel<F>` stores the parsed/runtime model and metadata (input/output names/shapes).
`SKOnnxPredictor<F>` wraps it and implements `SKRegressorPredictor<F>` and, when the output
is a class matrix, `SKClassifierPredictor<F>` (labels via argmax over the class axis). A
single output vector is a regression; a two-dimensional output with a class metadata is a
classification. Ambiguity returns `UnsupportedRepresentation`.

### Decision 3 — Input/output mapping

The input is a dense `ArrayView2<F>`; the predictor converts to the model dtype/shape,
runs inference and copies outputs back into ndarray. Non-finite inputs are rejected before
inference. Batch size is the row count; dynamic batch dimensions are supported when the
runtime allows.

### Decision 4 — Validation

Validate model bytes, input width/dtype, output count/shape and operator support before
returning a predictor. Errors: `Conversion`, `ShapeMismatch`, `UnsupportedOperation`,
`NonFiniteValue`.

### Decision 5 — File layout

```
crates/sciencekit_onnx/src/import/
  mod.rs, model.rs, predictor.rs, runtime.rs, interpreter.rs, metadata.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Tract MSRV/transitive dependencies** → spike first; fallback interpreter.
- **Operator coverage** → documented subset; unsupported ops explicit.
- **Output semantics ambiguity** → metadata plus shape rules; ambiguous models rejected.
- **Runtime dependency size** → feature-gated; the crate remains optional.
- **Dtype conversion** → explicit conversion with tests.

## Migration Plan

Additive: import module in `sciencekit_onnx`. No existing API changes.

## Open Questions

None blocking; the runtime choice is resolved by the spike.
