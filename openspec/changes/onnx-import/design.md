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

### Decision 1 — One shared ONNX IR, one interpreter

`serde-onnx` (pinned at revision `0c54592`, MSRV 1.85) is the single ONNX IR for both
directions: the exporter now assembles `serde_onnx::ir::Model` instead of hand-declared
prost messages, and the importer decodes the same type. `GraphBuilder`, the initializer
vocabulary and the typed `ai.onnx.ml` operators come from that crate, so export and import
can never drift apart.

A spike evaluated `tract-onnx 0.22` as the runtime. It builds on MSRV 1.85, but it brings
its own IR and roughly forty transitive crates while the library only needs the operators
its own exporters emit. The change therefore ships a documented interpreter over the
shared IR. The subset is explicit and advertised as data:
`SK_ONNX_SUPPORTED_OPERATORS` (Abs, Add, ArgMax, ArgMin, Cast, Clip, Concat, Constant, Div,
Equal, Exp, Flatten, Gather, Greater, Identity, Less, MatMul, Mul, Neg, Pow, ReduceMean,
ReduceSum, Reshape, Sigmoid, Softmax, Squeeze, Sub, Tanh, TopK, Transpose, Unsqueeze) and
`SK_ONNX_SUPPORTED_ML_OPERATORS` (`TreeEnsembleClassifier`, `TreeEnsembleRegressor`).

The pinned revision rejects a scalar attribute whose value equals the protobuf default
because it decides field presence from the encoded value. Legal ONNX graphs therefore use
`axis = 0` and `keepdims = 0`. `import/decode.rs` substitutes a sentinel before decoding and
restores the zero afterwards, so the decoded value is exactly the shared IR.

### Decision 2 — Model wrapper and predictor

`SKOnnxModel` (not generic) stores the encoded bytes plus the decoded shared IR; its
`from_bytes`/`from_file` constructors validate the model. `SKOnnxPredictor<F>` owns the
model and the analysed interface. The scalar parameter belongs to the predictor, not the
model: the IR is dtype-agnostic.
`SKOnnxPredictor<F>` implements both `SKRegressorPredictor<F>` and
`SKClassifierPredictor<F>`; a method whose head the model does not declare returns
`UnsupportedOperation` instead of guessing. A single float output is a regression (one
column for `predict`, any width for `predict_arrays`); an integer label plus a float matrix
is a classification; an integer-only output is a labels-only head. Any other combination is
`UnsupportedOutput`.

### Decision 3 — Input/output mapping

The input is a dense `ArrayView2<F>`; the predictor converts to the model dtype/shape,
runs inference and copies outputs back into ndarray. Non-finite inputs are rejected before
inference. Batch size is the row count; dynamic batch dimensions are supported when the
runtime allows.

### Decision 4 — Validation

Validate model bytes, input width/dtype, output count/shape and operator support before
returning a predictor. Errors: `SKOnnxImportError::{InvalidModel, UnsupportedOpset,
UnsupportedOperator, MissingInitializer, ExternalData, ShapeMismatch, UnsupportedDtype,
UnsupportedOutput, InvalidOperator}` plus the central `SKError` variants.

Numerical proof is part of the contract: every exported fixture is imported and compared
against both the original Rust prediction and `onnxruntime` on the same input. The harness
reports runtime matched/total and import matched/total on the 36 fixtures.

### Decision 5 — File layout

```
crates/sciencekit_onnx/src/import/
  mod.rs, decode.rs, error.rs, value.rs, model.rs, interface.rs,
  interpreter.rs, dispatch.rs, attributes.rs, initializers.rs,
  elementwise.rs, matrix.rs, reductions.rs, shape_ops.rs, selection.rs,
  tree_ensemble.rs, tree_walk.rs, predictor*.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Tract MSRV/transitive dependencies** → rejected in favour of the shared-IR interpreter.
- **The pinned serde-onnx decoder rejects zero-valued attributes** → a documented sentinel
  shim in `import/decode.rs`; remove it when the pinned revision is bumped.
- **Operator coverage** → documented subset; unsupported ops explicit.
- **Output semantics ambiguity** → metadata plus shape rules; ambiguous models rejected.
- **Runtime dependency size** → feature-gated; the crate remains optional.
- **Dtype conversion** → explicit conversion with tests.

## Migration Plan

Additive: import module in `sciencekit_onnx`. No existing API changes.

## Open Questions

None blocking; the runtime choice is resolved by the spike.
