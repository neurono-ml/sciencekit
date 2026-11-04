//! Logistic regression and its two estimators (binary and multiclass).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod builder;
pub mod core_implementation;
pub mod model;
pub(crate) mod streaming;
pub(crate) mod validation;

pub use builder::SKLogisticRegressionBuilder;
pub use core_implementation::SKLogisticRegression;
pub use model::SKLogisticRegressionModel;

#[cfg(test)]
mod logistic_regression_tests;
#[cfg(test)]
mod sparse_streaming_tests;
#[cfg(test)]
mod streaming_tests;
