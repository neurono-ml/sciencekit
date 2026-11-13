//! The shared coordinate-descent substrate for the L1-regularized linear models
//! (spec `coordinate-descent-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod access;
mod accumulator;
pub mod config;
mod core;
pub(crate) mod export_helpers;
mod fit;
mod screening;
mod solution;
mod solver;
mod sparse_accumulation;
mod streaming;

pub(crate) use config::SKCoordinateDescentConfig;
pub use config::{SKIntoAlphas, SKSelection};
pub(crate) use core::SKCoordinateDescentModelCore;
pub(crate) use fit::sk_fit_coordinate_descent;
pub(crate) use streaming::sk_fit_streaming;

#[cfg(test)]
mod accumulator_tests;
#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod screening_tests;
#[cfg(test)]
mod solver_tests;
#[cfg(test)]
mod sparse_streaming_tests;
#[cfg(test)]
pub(crate) mod test_datasets;
