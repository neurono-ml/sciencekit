# Design

## Context

See `proposal.md` — Why. `gpu-compute-contracts` defines the contracts; OpenCL/CUDA validate
them. ROCm uses HIP (`hip-sys`/runtime spike) behind `gpu-rocm`. Default builds do not
require ROCm.

## Goals / Non-Goals

**Goals:**
- A ROCm backend with CPU parity.
- Deterministic device selection and fallback.
- No ROCm dependency in the default build.

**Non-Goals:**
- OpenCL/CUDA/Metal, multi-GPU, memory pooling.
- rocBLAS/hipBLAS or moving heavy BLAS/SVD off CPU.

## Decisions

### Decision 1 — HIP bindings and runtime

Use `hip-sys` or a maintained HIP wrapper (spike) behind `gpu-rocm`; runtime requires ROCm.
Kernels are compiled/cached per architecture (HIPRTC or offline code objects).

### Decision 2 — Kernel set

Pairwise distance, GEMM, elementwise and tree prediction, mirroring OpenCL/CUDA. Each
kernel returns a device buffer and is validated against the CPU kernel.

### Decision 3 — Buffers and streams

`SKDeviceBuffer` wraps a HIP buffer with dtype/shape/capacity. Upload/download use stream
copies with events; the safe default waits before returning. Transfer bytes are recorded.

### Decision 4 — Device selection and fallback

Enumerate devices, filter by architecture/kernel availability, sort by device index, pick
the first. Feature off/no runtime/incompatible device/kernel failure → CPU fallback with a
reason.

### Decision 5 — File layout

```
crates/sciencekit_gpu_rocm/src/
  lib.rs
  device/{mod.rs, enumerate.rs, selection.rs, *_tests.rs}
  buffer/{mod.rs, transfer.rs, *_tests.rs}
  kernel/{mod.rs, pairwise.rs, gemm.rs, elementwise.rs, tree.rs, compile.rs, *_tests.rs}
  backend/{mod.rs, implementation.rs, *_tests.rs}
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **HIP binding maturity/MSRV** → spike and feature gate.
- **ROCm installation variance** → hardware-gated tests and explicit fallback.
- **Compilation/toolchain requirements** → prefer HIPRTC/offline code objects with clear
  errors.
- **Transfer overhead** → document break-even sizes.

## Migration Plan

Additive: new crate and feature. No existing API changes.

## Open Questions

None blocking; the HIP binding choice is the spike task.
