//! Device discovery and the compute-backend contract
//! (spec `gpu-compute-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod discovery;

#[cfg(test)]
pub(crate) mod mock_device;

pub use discovery::SKDeviceDiscovery;
