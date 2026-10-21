//! Elastic net regression (spec `elastic-net`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod builder;
pub mod core_implementation;
pub mod model;

pub use builder::SKElasticNetBuilder;
pub use core_implementation::SKElasticNet;
pub use model::SKElasticNetModel;

#[cfg(test)]
mod elastic_net_tests;
#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod sparse_tests;
#[cfg(test)]
mod thread_tests;
