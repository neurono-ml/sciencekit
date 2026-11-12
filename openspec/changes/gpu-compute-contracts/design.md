# Design

## Context

See `proposal.md` — Why. `sciencekit_common` owns `SKBackendKind`, `SKExecutionMode`,
`SKExecutionContext` and the observability spans. `sciencekit_math` owns the host-array
`SKMathBackend`. GPU work needs a different abstraction because device memory is not
`ArrayView2` and transfers are explicit.

## Goals / Non-Goals

**Goals:**
- One device-compute contract shared by OpenCL, CUDA and ROCm.
- Explicit CPU fallback and honest observability.
- No heavy GPU dependency in the default build.

**Non-Goals:**
- Concrete backends and kernels (next three changes).
- Metal, multi-GPU collectives, memory pooling.
- Moving heavy BLAS/SVD to GPU (stays on CPU per the wave plan).

## Decisions

### Decision 1 — `sciencekit_gpu` owns the device contracts

The crate defines:

```rust
pub struct SKDeviceInfo { pub kind: SKBackendKind, pub name: String, pub memory_bytes: u64, pub capabilities: SKDeviceCapabilities }
pub struct SKDeviceBuffer<F> { ... }
pub trait SKComputeBackend<F: SKFloat> {
    type Error: From<SKError>;
    fn devices(&self) -> Result<Vec<SKDeviceInfo>, Self::Error>;
    fn upload(&self, host: ArrayView2<'_, F>) -> Result<SKDeviceBuffer<F>, Self::Error>;
    fn download(&self, buffer: &SKDeviceBuffer<F>) -> Result<Array2<F>, Self::Error>;
    fn dispatch(&self, kernel: SKGpuKernel, args: &[SKDeviceBufferRef<'_, F>]) -> Result<SKDeviceBuffer<F>, Self::Error>;
}
```

`SKGpuKernel` is a closed enum (`PairwiseDistance`, `Gemm`, `Elementwise`, `TreePredict`).
The concrete backends implement the trait; `sciencekit_math` keeps the CPU `SKMathBackend`.

### Decision 2 — Extend `SKBackendKind`

Add `OpenCl`, `Cuda`, `Rocm`. The enum is not `#[non_exhaustive]`; every exhaustive match in
the workspace is updated. This is a breaking change for external exhaustive matches, noted
in the proposal. The display names are `opencl`, `cuda`, `rocm`.

### Decision 3 — Device selection

`SKDeviceSelection` is deterministic:

```
configured/available kinds → filter by required kernel → sort by (kind priority, device index)
→ first device; otherwise CPU fallback
```

`Automatic` prefers the explicitly configured GPU kind, then OpenCL (vendor-transparent),
then CUDA, then ROCm, then CPU. A requested GPU kind without a device falls back to CPU and
records the reason; an explicit `InProcessSynchronous` plan never selects a GPU.

### Decision 4 — Buffer and transfer semantics

Buffers carry dtype, shape and capacity. Upload copies from a host view; download copies
into an owned `Array2`. A mapped/zero-copy path is optional per backend and returns a
borrowed view only while the mapping is alive; the safe default is explicit transfer.

### Decision 5 — Planning and observability

`SKExecutionContext` gains an optional device inventory (populated by the backend). The
resolver remains pure given that inventory. Spans record the selected kind and fallback
reason via `SKOperationAttributes`.

### Decision 6 — Feature flags

`gpu-opencl`, `gpu-cuda`, `gpu-rocm` are off by default and only enable the corresponding
backend crate. The foundation crate itself has no GPU dependency; it only defines contracts
and the CPU fallback.

### Decision 7 — File layout

```
crates/sciencekit_gpu/src/
  lib.rs
  device/{mod.rs, info.rs, selection.rs, *_tests.rs}
  buffer/{mod.rs, device_buffer.rs, transfer.rs, *_tests.rs}
  kernel/{mod.rs, kind.rs, dispatch.rs, *_tests.rs}
  backend/{mod.rs, contract.rs, *_tests.rs}
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Adding enum variants breaks exhaustive matches** → workspace-wide update and a
  compatibility note; the enum is closed by design.
- **Device inventories can be expensive** → discovery is explicit and cacheable by the
  backend; planning takes the inventory as input and stays pure.
- **Transfer costs can erase GPU gains** → the kernel contract makes transfers explicit;
  each backend measures and documents the break-even size.
- **No backend in the foundation** → tests use a mock backend.

## Migration Plan

Additive crate plus enum variants. Update matches in the same change; no runtime behavior
changes on CPU-only builds.

## Open Questions

None blocking. Metal and multi-GPU collectives are future changes.
