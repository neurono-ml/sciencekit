//! Ridge regression (spec `ridge-regression`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod builder;
pub mod core_implementation;
pub mod model;

mod fitting_logic;
mod gram;
mod solving;
mod sparse;
mod streaming;

pub use builder::{SKIntoRidgeAlphas, SKRidgeBuilder, SKRidgeSolver};
pub use core_implementation::SKRidge;
pub use model::SKRidgeModel;

#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod partial_fit_tests;
#[cfg(test)]
mod ridge_tests;
#[cfg(test)]
mod solver_tests;
#[cfg(test)]
mod sparse_prediction_tests;
#[cfg(test)]
mod sparse_test_datasets;
#[cfg(test)]
mod sparse_tests;
#[cfg(test)]
mod streaming_test_datasets;
#[cfg(test)]
mod streaming_tests;
#[cfg(test)]
mod test_datasets;
#[cfg(test)]
mod thread_tests;
