//! Generic bootstrap-aggregating (bagging) classifiers and regressors
//! (spec `bagging-ensembles`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod config;
mod error;

pub use error::{SKBaggingError, SKBaseEstimatorError};

#[cfg(test)]
mod bagging_error_tests;
