//! `sciencekit_gpu` — the device-compute contracts shared by the OpenCL, CUDA
//! and ROCm backends (spec `gpu-compute-contracts`).
//!
//! The crate is a **contracts-only foundation**: it describes devices,
//! host↔device buffers, kernel dispatch and the deterministic CPU fallback, and
//! it pulls in no GPU runtime. Concrete backends live behind the `gpu-opencl`,
//! `gpu-cuda` and `gpu-rocm` features in their own crates; until one is
//! present, planning records the CPU fallback instead of guessing.
//!
//! The device vocabulary (`SKDeviceInfo`, `SKDeviceCapabilities`,
//! `SKGpuKernel`, `SKFallbackReason`) is defined in `sciencekit_common` so
//! execution planning can consume an inventory without depending on this crate;
//! this crate re-exports it as the public device contract.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod backend;
pub mod buffer;
pub mod device;
