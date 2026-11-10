//! The device-compute vocabulary shared by execution planning and the
//! `sciencekit_gpu` contracts (spec `gpu-compute-contracts`).
//!
//! The vocabulary lives here, in the crate that resolves execution plans, so
//! planning consumes a device inventory without depending on a GPU runtime or
//! on a circular dependency with `sciencekit_gpu`. That crate re-exports these
//! types as the public device contract its backends implement.

mod capabilities;
mod info;

pub use capabilities::SKDeviceCapabilities;
pub use info::SKDeviceInfo;

#[cfg(test)]
mod device_tests;
