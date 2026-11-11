//! Generic bootstrap-aggregating (bagging) classifiers and regressors
//! (spec `bagging-ensembles`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

#[cfg(test)]
mod bagging_builder_tests;
#[cfg(test)]
mod bagging_error_tests;

pub mod builder;

mod classifier;
mod config;
mod error;
mod regressor;
pub use builder::{SKBaggingClassifierBuilder, SKBaggingRegressorBuilder};
pub use classifier::SKBaggingClassifier;
pub use error::{SKBaggingError, SKBaseEstimatorError};
pub use regressor::SKBaggingRegressor;
