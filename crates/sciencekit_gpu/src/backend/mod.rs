//! Device discovery and the compute-backend contract
//! (spec `gpu-compute-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod contract;
mod discovery;

pub use contract::SKComputeBackend;
pub use discovery::SKDeviceDiscovery;

#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
pub(crate) mod mock_device;
