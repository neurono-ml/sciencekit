# Proposal

## Why

OpenCL is vendor-transparent but CUDA remains the highest-performance path on NVIDIA and is
explicitly requested. The device contracts and OpenCL backend prove the kernel/transfer
surface; CUDA can implement the same contracts behind its own feature without changing the
CPU build.

## What Changes

- New `sciencekit_gpu_cuda` crate implementing `SKComputeBackend<F>` over `cudarc` (spike
  version/MSRV) and the CUDA driver/runtime.
- Device enumeration, capability reporting and deterministic selection among CUDA devices.
- Kernels: pairwise distance, GEMM, elementwise and tree prediction; NVRTC-compiled or
  precompiled PTX kernels, cached per device.
- Host↔device buffers, asynchronous copies/events and explicit synchronization.
- CPU fallback when the feature, driver, device or kernel is unavailable.
- Feature `gpu-cuda`, off by default; no CUDA toolkit required to build the default crates.
- Numerical parity with CPU kernels and hardware-gated tests (skip cleanly without a GPU).
- Validation: missing driver, unsupported compute capability, kernel compile failure,
  transfer failure and out-of-memory.
- Explicit non-goals: OpenCL/ROCm, Metal, multi-GPU collectives, cuBLAS/cuDNN integration.

## Capabilities

### New Capabilities

- `cuda-backend`: CUDA device discovery, buffers, kernel compilation/dispatch, CPU fallback
  and parity guarantees.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_gpu_cuda` workspace member and feature wiring.
- **APIs:** additive; selected through `SKBackendKind::Cuda`.
- **Dependencies:** `cudarc` (spike) and the CUDA driver at runtime; optional.
- **Downstream:** gives NVIDIA users a native path and validates the device contracts on a
  second backend.
- **Acceptance criteria:** CPU parity, fallback, deterministic selection, concurrency and
  hardware-gated tests.
