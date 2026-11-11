//! Random-forest classifier and regressor estimators, builders and fitted
//! models (spec `random-forests`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier;
mod classifier_builder;
mod classifier_fit;
mod classifier_model;
mod classifier_predict;
mod execution;
mod input_validation;
mod regressor;
mod regressor_builder;
mod regressor_fit;
mod regressor_model;
mod regressor_predict;
mod validation;

pub use crate::bootstrap::SKMaximumSamples;
pub use classifier::SKRandomForestClassifier;
pub use classifier_builder::SKRandomForestClassifierBuilder;
pub use classifier_model::SKRandomForestClassifierModel;
pub use regressor::SKRandomForestRegressor;
pub use regressor_builder::SKRandomForestRegressorBuilder;
pub use regressor_model::SKRandomForestRegressorModel;

#[cfg(test)]
mod forest_builder_tests;
#[cfg(test)]
mod forest_classifier_tests;
#[cfg(test)]
mod forest_fit_tests;
#[cfg(test)]
mod forest_regressor_tests;
