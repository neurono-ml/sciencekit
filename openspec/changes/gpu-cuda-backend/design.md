# Design

## Context

See `proposal.md` — Why. `gpu-compute-contracts` defines the device contracts; the OpenCL
backend validates them. CUDA uses `cudarc` (spike version; latest 0.19.10, MSRV unknown) and
the driver/runtime. No CUDA toolkit is required to build the default workspace.

## Goals / Non-Goals

**Goals:**
- A CUDA backend with CPU parity.
- Deterministic device selection and explicit fallback.
- No CUDA dependency in the default build.

**Non-Goals:**
- OpenCL/ROCm/Metal, multi-GPU, memory pooling.
- cuBLAS/cuDNN or moving heavy BLAS/SVD off CPU.

## Decisions

### Decision 1 — `cudarc` + driver/runtime

Use `cudarc` behind `gpu-cuda`; runtime requires a CUDA driver. Device programs are
compiled with NVRTC (when available) or precompiled PTX per compute capability, cached in
the backend.

### Decision 2 — Kernel set

Pairwise distance, GEMM, elementwise and tree prediction, mirroring the OpenCL kernel
catalogue. Each kernel returns a device buffer and is validated against the CPU kernel.

### Decision 3 — Buffers and streams

`SKDeviceBuffer` wraps a CUDA buffer with dtype/shape/capacity. Upload/download use
asynchronous copies on a stream with event synchronization; the safe default waits before
returning. Transfer bytes are recorded for observability.

### Decision 4 — Device selection and fallback

Enumerate devices, filter by compute capability/kernel availability, sort by device index,
pick the first. Feature off/no driver/incompatible device/kernel failure → CPU fallback
with a reason.

### Decision 5 — File layout

```
crates/sciencekit_gpu_cuda/src/
  lib.rs
  device/{mod.rs, enumerate.rs, selection.rs, *_tests.rs}
  buffer/{mod.rs, transfer.rs, *_tests.rs}
  kernel/{mod.rs, pairwise.rs, gemm.rs, elementwise.rs, tree.rs, compile.rs, *_tests.rs}
  backend/{mod.rs, implementation.rs, *_tests.rs}
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **`cudarc` MSRV/driver variance** → spike and feature gate.
- **NVRTC availability** → fall back to precompiled PTX or CPU.
- **Transfer overhead** → document break-even sizes.
- **Hardware CI** → hardware-gated tests skip cleanly.

## Migration Plan

Additive: new crate and feature. No existing API changes.

## Open Questions

None blocking; the `cudarc` version is the spike task.
