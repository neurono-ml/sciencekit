//! Lasso regression (spec `lasso`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod builder;
pub mod core_implementation;
pub mod model;

pub use builder::SKLassoBuilder;
pub use core_implementation::SKLasso;
pub use model::SKLassoModel;

#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod lasso_tests;
#[cfg(test)]
mod sparse_tests;
#[cfg(test)]
mod streaming_tests;
#[cfg(test)]
mod thread_tests;
