//! The device-buffer and transfer contract (spec `gpu-compute-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod buffer_builder;
mod device_buffer;
mod transfer;

pub use buffer_builder::SKDeviceBufferBuilder;
pub use device_buffer::SKDeviceBuffer;
pub use transfer::SKDeviceTransferPolicy;

#[cfg(test)]
mod buffer_tests;
