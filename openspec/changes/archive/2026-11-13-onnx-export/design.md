# Design

## Context

See `proposal.md` — Why. `sciencekit_interop`/`sciencekit_common` supply the model state;
the model crates supply concrete types. ONNX needs its own crate because it depends on
those types and on a protobuf encoding. Key MSRV findings:

- `tract-onnx 0.22` has MSRV 1.85 and can validate exported graphs; `0.23` requires 1.91.
- `onnx-protobuf 0.2.3` uses the old `protobuf 3.4` line and is MPL-2.0.
- `axonml-onnx 0.6.2` declares MSRV 1.85.
- `prost 0.14` declares MSRV 1.85.

The protobuf choice is a spike task; the architecture does not depend on it.

## Goals / Non-Goals

**Goals:**
- A valid ONNX graph per supported model family.
- Stable metadata and dtype mapping.
- Export validation through a real runtime.

**Non-Goals:**
- ONNX import (separate change), native-type import, training, GPU and Python.
- Every exotic operator; unsupported families are explicit.
- Quantized models.

## Decisions

### Decision 0 — protobuf encoding: `prost` with declared messages (spike result)

The spike evaluated the three candidates under MSRV 1.85 and license policy
(Apache-2.0 project):

| Candidate | License | MSRV | Finding |
|---|---|---|---|
| `prost` 0.14.5 | Apache-2.0 | 1.85 | Already in the workspace lock (gRPC). Messages can be declared with `#[derive(prost::Message)]` directly, so no `protoc` or `prost-build` runs and no vendored `.proto` file is needed. |
| `onnx-protobuf` 0.2.3 | MPL-2.0 | — | Depends on the old `protobuf 3.4` line; the file-level copyleft is a licensing complication the project avoids. |
| `axonml-onnx` 0.6.2 | MIT OR Apache-2.0 | 1.85 | Admissible license, but pulls `axonml-core`, `axonml-nn`, `axonml-tensor` and `axonml-serialize` — a whole foreign ML framework for a pure encoder. |

**Decision:** `prost` 0.14 with hand-declared ONNX message structs. Evidence:
`cargo test -p sciencekit_onnx` compiles and passes on Rust 1.85, and
`onnx.checker.check_model` plus `onnxruntime` accept the produced bytes for the
identity graph (`validation/compare_with_onnxruntime.py`). The builder API stays
backend-agnostic: only `graph/proto.rs` knows about protobuf.

### Decision 1 — `sciencekit_onnx` owns the trait and the exporters

`SKToOnnx<F>` lives here, not in `sciencekit_common` (which must stay ONNX-free) and not in
each model crate (which would add a heavy dependency). Implementations in this crate are
allowed because the trait is local.

```rust
pub trait SKToOnnx<F: SKFloat> {
    fn to_onnx(&self) -> Result<SKOnnxModel, SKError>;
}
```

`SKOnnxModel` holds the serialized `ModelProto` bytes and the opset.

### Decision 2 — Graph builder over a protobuf backend

`SKOnnxGraphBuilder` exposes `input`, `initializer`, `node`, `output` and `metadata`.
The protobuf backend is chosen by the spike; the builder API is backend-agnostic. Opset 17
is the default (widely supported; includes the ONNX-ML tree ensemble ops used by trees).

### Decision 3 — Per-family operator mapping

- Preprocessing: `Sub`/`Div`/`Mul`/`Add`/`Clip`/`OneHot`/`MatMul` as appropriate.
- Linear/logistic: `MatMul`, `Add`, `Sigmoid`, `Softmax`, `ArgMax`.
- SVM: ONNX-ML `SVMClassifier`/`SVMRegressor` where the kernel is supported, or explicit
  support-vector expansion for linear/RBF.
- Trees/forests/boosting: ONNX-ML `TreeEnsembleClassifier`/`TreeEnsembleRegressor`.
- Neighbours: pairwise distance (`Sub`/`Mul`/`ReduceSum`/`Sqrt`), `TopK`, `Gather`,
  `ReduceMean`.
- Clustering: distances plus `ArgMin`.
- Decomposition: `Sub`, `MatMul`.
- Outlier: tree-ensemble path lengths or explicit tree expansion.
- Pipelines: compose stage subgraphs in order.

A coverage matrix in the crate docs records supported families and operators.

### Decision 4 — Metadata and dtype

Model metadata includes producer (`sciencekit`), estimator name, format version, scalar
dtype and class table. `f32 → FLOAT`, `f64 → DOUBLE`. Integer tensors map to `INT64` and
bools to `BOOL`.

### Decision 5 — Validation

Validate state completeness/shapes before building. After building, load the bytes with a
compatible runtime (`tract-onnx 0.22` as a dev-dependency) and compare predictions on
fixtures. A missing tensor or unsupported operator returns a structured error.

### Decision 6 — File layout

```
crates/sciencekit_onnx/src/
  lib.rs
  graph/{mod.rs, builder.rs, proto.rs, opset.rs, metadata.rs, *_tests.rs}
  export/{mod.rs, preprocessing.rs, linear.rs, svm.rs, tree.rs, neighbors.rs,
          cluster.rs, decomposition.rs, outlier.rs, pipeline.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Protobuf backend choice** → spike before implementation; the builder isolates it.
- **ONNX operator coverage is broad** → a documented matrix and explicit
  `UnsupportedOperation` beats a silently wrong graph.
- **SVM/tree op semantics differ across runtimes** → validate with tract and a second
  checker if available.
- **f32/f64 parity** → per-dtype fixtures.
- **Large graphs** → initializers keep tensors out of nodes; document graph size.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking; the protobuf backend is resolved by the spike task.
