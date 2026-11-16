## Status

**Deferred — see ADR #408.** No ROCm runtime is available to this project: the only
AMD-capable node exposes `/dev/kfd` and `/dev/dri` but `/opt/rocm` is empty and neither
`rocminfo` nor `hipcc` exists, so no agent can be enumerated and no HIP kernel can be
compiled or verified. The plan below is preserved as design material, not as committed
near-term work. Tracking issue #410 states what a future contribution must satisfy before
this change is reactivated.

# Proposal

## Why

ROCm is the AMD performance path and is explicitly requested. The device contracts and the
two previous backends prove the surface; ROCm/HIP can implement the same kernels behind its
own feature, keeping the CPU and default builds clean.

## What Changes

- New `sciencekit_gpu_rocm` crate implementing `SKComputeBackend<F>` over ROCm/HIP
  (`hip-sys`/HIP runtime spike).
- Device enumeration, capability reporting and deterministic selection among ROCm devices.
- Kernels: pairwise distance, GEMM, elementwise and tree prediction, compiled/cached per
  device.
- Host↔device buffers, stream copies/events and explicit synchronization.
- CPU fallback when the feature, runtime, device or kernel is unavailable.
- Feature `gpu-rocm`, off by default; no ROCm toolkit required for the default build.
- Numerical parity with CPU kernels and hardware-gated tests (skip cleanly without a GPU).
- Validation: missing runtime, unsupported architecture, kernel compile failure, transfer
  failure and out-of-memory.
- Explicit non-goals: OpenCL/CUDA/Metal, multi-GPU, rocBLAS/hipBLAS integration.

## Capabilities

### New Capabilities

- `rocm-backend`: ROCm device discovery, buffers, kernel compilation/dispatch, CPU fallback
  and parity guarantees.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_gpu_rocm` workspace member and feature wiring.
- **APIs:** additive; selected through `SKBackendKind::Rocm`.
- **Dependencies:** HIP bindings (spike) and the ROCm runtime; optional.
- **Downstream:** gives AMD users a native path and completes the requested GPU family.
- **Acceptance criteria:** CPU parity, fallback, deterministic selection, concurrency and
  hardware-gated tests.
