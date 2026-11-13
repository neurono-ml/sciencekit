# Proposal

## Why

OpenCL is the first GPU target because it is vendor-transparent: one ICD-based backend can
run on NVIDIA, AMD, Intel and CPU OpenCL devices. The wave plan locks OpenCL 3.0 first. The
foundation contracts exist in `gpu-compute-contracts`; this change delivers the first real
device backend and proves the kernel/transfer surface on hardware.

## What Changes

- New `sciencekit_gpu_opencl` crate implementing `SKComputeBackend<F>` over OpenCL 3.0 via
  an ICD loader (`opencl3`/`ocl` spike).
- Device enumeration (platforms/devices), capability reporting and deterministic selection.
- Kernels: pairwise distance (Euclidean/squared/Manhattan/Chebyshev), GEMM, elementwise
  transform/combine and tree prediction.
- Host↔device buffers, optional mapped/zero-copy path and explicit transfer accounting.
- CPU fallback when no device/platform/kernel is available; feature `gpu-opencl`, off by
  default.
- Numerical parity tests against the CPU kernels on small and large inputs; hardware-gated
  CI tests plus a POCL path.
- Validation: missing platform, unsupported dtype, kernel compile failure, transfer
  failure and out-of-memory.
- Explicit non-goals: CUDA/ROCm (separate changes), Metal, multi-GPU collectives, moving
  heavy BLAS/SVD off CPU, GPU memory pooling.

## Capabilities

### New Capabilities

- `opencl-backend`: OpenCL device discovery, buffers, kernel compilation/dispatch, CPU
  fallback and parity guarantees.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_gpu_opencl` workspace member and feature wiring.
- **APIs:** additive; selected through `SKBackendKind::OpenCl`.
- **Dependencies:** `opencl3` or `ocl` (spike), OpenCL ICD loader at runtime; optional.
- **Downstream:** gives the library its first vendor-transparent GPU path and validates
  the device contracts.
- **Acceptance criteria:** CPU parity, fallback, deterministic device selection, large/small
  kernels, concurrency and hardware-gated tests.
