//! The `accelerators` family: device-compute contracts and the OpenCL backend.
//!
//! Every item here resolves to `sciencekit_gpu`. The device vocabulary
//! (`SKDeviceInfo`, `SKDeviceCapabilities`, `SKGpuKernel`, `SKFallbackReason`)
//! is owned by `sciencekit_common`, so the names here and at the umbrella root
//! are the same types.

pub use sciencekit_gpu::{
    SK_HOST_FALLBACK_BACKEND, SKDeviceCapabilities, SKDeviceContext, SKDeviceInfo,
    SKDevicePreference, SKFallbackReason, sk_clear_device_inventory, sk_register_device_inventory,
    sk_registered_device_inventory, sk_resolve_operation_backend, sk_select_device,
};
pub use sciencekit_gpu::{SKComputeBackend, SKDeviceDiscovery};
pub use sciencekit_gpu::{SKDeviceBuffer, SKDeviceBufferBuilder, SKDeviceTransferPolicy};
pub use sciencekit_gpu::{SKDeviceBufferRef, SKGpuKernel};
pub use sciencekit_gpu::{SKOpenClBackend, SKOpenClBackendBuilder, SKOpenClTransferReport};
pub use sciencekit_gpu::{backend, buffer, device, kernel, opencl};
