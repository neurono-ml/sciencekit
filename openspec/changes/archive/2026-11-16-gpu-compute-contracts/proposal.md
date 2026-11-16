# Proposal

## Why

The library is CPU-only: `SKBackendKind` names three CPU backends and `SKMathBackend` is a
host-array trait. GPU acceleration needs a device buffer/context contract, explicit backend
kinds for OpenCL/CUDA/ROCm, device discovery and a planning rule that falls back to CPU
when no device or kernel is available. Without that foundation, each GPU backend would
invent its own dispatch surface.

## What Changes

- New `sciencekit_gpu` crate with the device-compute contracts: `SKComputeBackend<F>`,
  `SKDeviceBuffer<F>`, `SKDeviceInfo`, `SKDeviceKind` and a kernel-dispatch surface.
- Extend `SKBackendKind` with `OpenCl`, `Cuda` and `Rocm`; update every exhaustive match
  in backend selection, observability and execution planning.
- Host↔device transfers (upload/download) and an optional zero-copy/mapped-buffer path.
- Multi-device enumeration, capability reporting and a deterministic device-selection rule.
- Execution planning: `Automatic` selects a GPU only when a compatible device and kernel
  exist; otherwise it falls back to the CPU backend and records the reason.
- Feature flags `gpu-opencl`, `gpu-cuda`, `gpu-rocm` (all off by default); no concrete
  backend implementation in this change.
- Validation: unavailable device, unsupported dtype, kernel not implemented, transfer
  failure and buffer shape mismatch.
- Explicit non-goals: concrete OpenCL/CUDA/ROCm code, GPU kernels, Metal, GPU memory
  pooling and multi-GPU collectives.

## Capabilities

### New Capabilities

- `gpu-compute-contracts`: backend kinds, device discovery, device buffers, kernel
  dispatch, GPU-aware planning and CPU fallback.

### Modified Capabilities

- `backend-selection`: the closed backend enum gains the GPU variants and every operation
  records the selected GPU kind.
- `execution-decision`: automatic planning may select a GPU backend when available and
  otherwise falls back to CPU.

## Impact

- **Code:** new `crates/sciencekit_gpu` workspace member; `sciencekit_common` backend enum
  and planning updates.
- **APIs:** additive crate; **BREAKING** for exhaustive matches over `SKBackendKind`
  (inside the workspace only).
- **Dependencies:** none in the foundation; the concrete backends add their own optional
  crates.
- **Downstream:** unblocks `gpu-opencl-backend`, `gpu-cuda-backend` and
  `gpu-rocm-backend`.
- **Acceptance criteria:** device enumeration is deterministic, CPU fallback is explicit,
  buffers round-trip, planning is pure and concurrent calls agree.
