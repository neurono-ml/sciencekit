//! Random-forest classifier and regressor estimators, builders and fitted
//! models (spec `random-forests`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier;
mod classifier_builder;
mod regressor;
mod regressor_builder;
mod validation;

pub use crate::bootstrap::SKMaximumSamples;
pub use classifier::SKRandomForestClassifier;
pub use classifier_builder::SKRandomForestClassifierBuilder;
pub use regressor::SKRandomForestRegressor;
pub use regressor_builder::SKRandomForestRegressorBuilder;

#[cfg(test)]
mod forest_builder_tests;
