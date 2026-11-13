//! The OpenCL compute backend (spec `opencl-backend`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; the builder, the backend, its operations and their tests live in
//! their own files.

mod builder;
mod implementation;

mod contract_impl;
#[cfg(feature = "opencl")]
mod operations;
#[cfg(not(feature = "opencl"))]
mod unavailable_operations;

pub use builder::SKOpenClBackendBuilder;
pub use implementation::{SKOpenClBackend, SKOpenClTransferReport};

#[cfg(all(test, feature = "opencl"))]
mod concurrency_tests;
#[cfg(test)]
mod fallback_tests;
#[cfg(all(test, feature = "opencl"))]
mod observability_tests;
