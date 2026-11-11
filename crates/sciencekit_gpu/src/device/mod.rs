//! The device contract surface (spec `gpu-compute-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file. The device
//! vocabulary is defined in `sciencekit_common` — where execution planning
//! consumes it — and re-exported here as this crate's public contract.

pub use sciencekit_common::device::{
    SK_HOST_FALLBACK_BACKEND, SKDeviceCapabilities, SKDeviceContext, SKDeviceInfo,
    SKDevicePreference, SKFallbackReason, sk_clear_device_inventory, sk_register_device_inventory,
    sk_registered_device_inventory, sk_resolve_operation_backend, sk_select_device,
};

#[cfg(test)]
mod device_tests;
#[cfg(test)]
mod selection_tests;
