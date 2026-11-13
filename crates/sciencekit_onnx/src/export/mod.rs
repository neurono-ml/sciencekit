//! The ONNX export contract and the state-to-graph dispatcher.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

#[cfg(test)]
mod contract_tests;
#[cfg(test)]
mod coverage_tests;
#[cfg(test)]
mod linear_tests;
#[cfg(test)]
mod preprocessing_tests;
#[cfg(test)]
mod remaining_tests;
#[cfg(test)]
mod tree_tests;

mod cluster;
mod contract;
mod coverage;
mod decomposition;
mod dispatcher;
mod helpers;
mod linear;
mod metadata;
mod neighbors;
mod preprocessing;
mod stage;
mod tree;
pub use contract::SKToOnnx;
pub use coverage::{SKOnnxFamilyCoverage, sk_onnx_coverage, sk_onnx_supported_families};
pub use dispatcher::sk_export_state_to_onnx;
