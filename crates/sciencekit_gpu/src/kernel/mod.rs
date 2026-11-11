//! The kernel catalogue and its dispatch contract (spec
//! `gpu-compute-contracts`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod dispatch;
mod kind;

pub use dispatch::SKDeviceBufferRef;
pub use kind::SKGpuKernel;

#[cfg(test)]
mod kernel_tests;
