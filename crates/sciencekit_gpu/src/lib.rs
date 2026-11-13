//! `sciencekit_gpu` — the device-compute contracts shared by the OpenCL, CUDA
//! and ROCm backends (spec `gpu-compute-contracts`), plus the concrete OpenCL
//! backend (spec `opencl-backend`).
//!
//! The contract surface describes devices, host↔device buffers, kernel dispatch
//! and the deterministic CPU fallback, and pulls in no GPU runtime by default.
//! The concrete OpenCL backend lives behind the `opencl` feature: with the
//! feature off the backend still exists as an honest fallback that calls no
//! OpenCL symbol, and with it on the backend enumerates the ICD loader's
//! devices, compiles its kernels on the selected device and moves buffers
//! explicitly.
//!
//! The device vocabulary (`SKDeviceInfo`, `SKDeviceCapabilities`,
//! `SKGpuKernel`, `SKFallbackReason`) is defined in `sciencekit_common` so
//! execution planning can consume an inventory without depending on this crate;
//! this crate re-exports it as the public device contract.

#![cfg_attr(not(feature = "opencl"), forbid(unsafe_code))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod backend;
pub mod buffer;
pub mod device;
pub mod kernel;
pub mod opencl;

pub use backend::{SKComputeBackend, SKDeviceDiscovery};
pub use buffer::{SKDeviceBuffer, SKDeviceBufferBuilder, SKDeviceTransferPolicy};
pub use device::{
    SK_HOST_FALLBACK_BACKEND, SKDeviceCapabilities, SKDeviceContext, SKDeviceInfo,
    SKDevicePreference, SKFallbackReason, sk_clear_device_inventory, sk_register_device_inventory,
    sk_registered_device_inventory, sk_resolve_operation_backend, sk_select_device,
};
pub use kernel::{SKDeviceBufferRef, SKGpuKernel};
pub use opencl::{SKOpenClBackend, SKOpenClBackendBuilder, SKOpenClTransferReport};
