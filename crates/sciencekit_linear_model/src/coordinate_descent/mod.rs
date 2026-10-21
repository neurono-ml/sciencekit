//! The shared coordinate-descent substrate for the L1-regularized linear models
//! (spec `coordinate-descent-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod access;
pub mod config;
mod screening;
mod solution;
mod solver;

pub use config::{SKIntoAlphas, SKSelection};

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod screening_tests;
#[cfg(test)]
mod solver_tests;
#[cfg(test)]
mod test_datasets;
