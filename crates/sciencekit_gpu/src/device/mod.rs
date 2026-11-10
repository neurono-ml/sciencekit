//! The device contract surface (spec `gpu-compute-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file. The device
//! vocabulary is defined in `sciencekit_common` — where execution planning
//! consumes it — and re-exported here as this crate's public contract.

pub use sciencekit_common::device::{SKDeviceCapabilities, SKDeviceInfo};

#[cfg(test)]
mod device_tests;
