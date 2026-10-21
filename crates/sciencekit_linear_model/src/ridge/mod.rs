//! Ridge regression (spec `ridge-regression`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod builder;
pub mod core_implementation;
pub mod model;

mod gram;
mod solving;

pub use builder::{SKIntoRidgeAlphas, SKRidgeBuilder, SKRidgeSolver};
pub use core_implementation::SKRidge;
pub use model::SKRidgeModel;

#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod ridge_tests;
#[cfg(test)]
mod solver_tests;
#[cfg(test)]
mod test_datasets;
