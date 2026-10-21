//! Linear regression (spec `linear-regression`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod builder;
pub mod core_implementation;
pub mod model;
pub mod streaming;

mod gram;
mod solving;

#[cfg(test)]
mod linear_regression_tests;
#[cfg(test)]
mod partial_fit_tests;
#[cfg(test)]
mod streaming_tests;
#[cfg(test)]
mod test_datasets;
