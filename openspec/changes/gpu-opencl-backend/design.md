# Design

## Context

See `proposal.md` — Why. `gpu-compute-contracts` defines `SKComputeBackend`, device info,
buffers and the kernel enum. This crate implements them over OpenCL 3.0. The spike chooses
`opencl3` vs `ocl`; both wrap the ICD loader.

## Goals / Non-Goals

**Goals:**
- A working OpenCL backend with CPU parity.
- Deterministic device selection and honest fallback.
- No OpenCL dependency in the default build.

**Non-Goals:**
- CUDA/ROCm/Metal, multi-GPU, memory pooling.
- Moving heavy BLAS/SVD to GPU.
- Every algorithm kernel.

## Decisions

### Decision 1 — OpenCL 3.0 through the ICD loader

Use `opencl3` behind the `opencl` feature (re-exported by the `sciencekit_gpu_opencl`
package through `gpu-opencl`). Runtime requires an ICD loader installed; no vendor SDK is
linked at build time. Device programs are compiled from embedded OpenCL C sources once per
device and cached in the backend.

**Spike evidence** (RTX 3070 Ti Laptop GPU, `clinfo`: `Platform #0: NVIDIA CUDA`, device 0;
`libOpenCL.so.1` installed, no development symlink, no CL headers):

- `ocl` 0.19.7 was rejected: its link step runs `-lOpenCL` and fails with
  `cannot find -lOpenCL: No such file or directory`, because only the versioned runtime
  library is installed. It cannot be used without installing a development package.
- `opencl3` 0.12.4 (latest) was rejected for the workspace MSRV: `context.rs` uses
  `Vec::len` in a const context, which needs a newer Rust than 1.85.
- `opencl3` **0.12.0** (exact pin) compiles on Rust 1.85, resolves to `cl3` 0.13.2, whose
  dynamic loader opens `libOpenCL.so.1` at runtime. A spike kernel multiplied a host vector
  by 2 on the device and read back `[2.0, 4.0, 6.0, 8.0]` from
  `NVIDIA GeForce RTX 3070 Ti Laptop GPU`.

**Package layout.** The implementation lives in `sciencekit_gpu` under
`src/opencl/` behind the `opencl` feature, not in a separate implementation crate: the
contract traits (`SKComputeBackend`, `SKDeviceDiscovery`) are defined in `sciencekit_gpu`, so
a crate that depends on them cannot also be a dependency of `sciencekit_gpu` (package
cycle). `crates/sciencekit_gpu_opencl` is the planned vendor-facing package and feature name;
it re-exports the backend and enables `sciencekit_gpu/opencl`, exposing no second contract.

### Decision 2 — Kernel set

- Pairwise distance: Euclidean/squared/Manhattan/Chebyshev, one work item per pair or tiled
  rows; parity with `sciencekit_math` kernels.
- GEMM: tiled local-memory kernel for dense `f32`/`f64`.
- Elementwise: transform/combine over contiguous buffers.
- Tree prediction: per-row traversal with nodes/thresholds as buffers.

Unsupported kernels return `UnsupportedOperation`; the planner keeps their work on CPU.

### Decision 3 — Buffers and transfers

`SKDeviceBuffer` wraps an `opencl3` buffer with dtype/shape/capacity. Upload/download use
blocking or non-blocking reads with an event wait. A mapped-buffer path is optional and
documented; the safe default copies. Transfer time is recorded for observability.

### Decision 4 — Device selection and fallback

Enumerate platforms/devices, filter by required kernel, sort by `(kind priority, device
index)`, and pick the first. If the feature is off, no platform exists, the device is
incompatible or the kernel fails to compile, return the fallback reason and let planning
select CPU.

### Decision 5 — Observability and errors

Record device name, kind, kernel and transfer bytes in spans. Errors: `UnsupportedOperation`,
`ShapeMismatch`, `NonFiniteValue`, `Conversion` (build log) and I/O/device errors.

### Decision 6 — File layout

```
crates/sciencekit_gpu_opencl/src/
  lib.rs
  device/{mod.rs, enumerate.rs, selection.rs, *_tests.rs}
  buffer/{mod.rs, transfer.rs, *_tests.rs}
  kernel/{mod.rs, pairwise.rs, gemm.rs, elementwise.rs, tree.rs, compile.rs, *_tests.rs}
  backend/{mod.rs, implementation.rs, *_tests.rs}
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Vendor driver quality varies** → parity tests on the CI device and POCL; feature-gated
  hardware tests.
- **Kernel compilation at runtime** → cache programs per device; surface build logs.
- **Transfer overhead** → document break-even sizes and keep small work on CPU.
- **ICD loader absence** → explicit fallback, no panic.
- **Precision differences** → f32/f64 tolerance tests.

## Migration Plan

Additive: new crate and feature. No existing API changes.

## Open Questions

None blocking; the `opencl3` vs `ocl` choice is the spike task.
