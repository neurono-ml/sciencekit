//! The device-compute vocabulary shared by execution planning and the
//! `sciencekit_gpu` contracts (spec `gpu-compute-contracts`).
//!
//! The vocabulary lives here, in the crate that resolves execution plans, so
//! planning consumes a device inventory without depending on a GPU runtime or
//! on a circular dependency with `sciencekit_gpu`. That crate re-exports these
//! types as the public device contract its backends implement.

mod capabilities;
mod fallback;
mod info;
mod kernel;
mod selection;

pub use capabilities::SKDeviceCapabilities;
pub use fallback::{SK_HOST_FALLBACK_BACKEND, SKFallbackReason};
pub use info::SKDeviceInfo;
pub use kernel::SKGpuKernel;
pub use selection::{SKDevicePreference, sk_select_device};

#[cfg(test)]
mod device_tests;
