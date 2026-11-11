//! Generic bootstrap-aggregating (bagging) classifiers and regressors
//! (spec `bagging-ensembles`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod classifier;
mod config;
mod error;
mod fit;
mod model;
mod oob;
mod predict;
mod regressor;
mod sampling;
mod validation;

pub use builder::{SKBaggingClassifierBuilder, SKBaggingRegressorBuilder};
pub use classifier::SKBaggingClassifier;
pub use error::{SKBaggingError, SKBaseEstimatorError};
pub use model::{SKBaggingClassifierModel, SKBaggingRegressorModel};
pub use regressor::SKBaggingRegressor;

#[cfg(test)]
mod bagging_bounds_tests;
#[cfg(test)]
mod bagging_builder_tests;
#[cfg(test)]
mod bagging_classifier_tests;
#[cfg(test)]
mod bagging_error_tests;
#[cfg(test)]
mod bagging_fit_tests;
#[cfg(test)]
mod bagging_oob_tests;
#[cfg(test)]
mod bagging_regressor_tests;
#[cfg(test)]
mod bagging_routing_tests;
#[cfg(test)]
mod bagging_sampling_tests;
