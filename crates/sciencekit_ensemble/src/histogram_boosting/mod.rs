//! Histogram gradient boosting (spec `histogram-gradient-boosting`): quantile
//! binning, histogram split finding, leaf-wise growth, categorical partitions,
//! missing-value routing and the boosted classifier/regressor surface.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

pub mod binning;

mod classifier;
mod classifier_builder;
mod classifier_fit;
mod classifier_model;
mod classifier_predict;
mod config;
mod export;
mod export_helpers;
mod fit;
pub mod growth;
pub mod histogram;
mod loss_kind;
mod multiclass_loop;
mod regressor;
mod regressor_builder;
mod regressor_fit;
mod regressor_model;
mod regressor_predict;
mod scalar_loop;
mod stage;
mod validation;

pub use classifier::SKHistGradientBoostingClassifier;
pub use classifier_builder::SKHistGradientBoostingClassifierBuilder;
pub use classifier_model::SKHistGradientBoostingClassifierModel;
pub use loss_kind::SKHistGradientBoostingRegressionLoss;
pub use regressor::SKHistGradientBoostingRegressor;
pub use regressor_builder::SKHistGradientBoostingRegressorBuilder;
pub use regressor_model::SKHistGradientBoostingRegressorModel;

#[cfg(test)]
mod classifier_tests;
#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod early_stopping_tests;
#[cfg(test)]
mod multiclass_loop_tests;
#[cfg(test)]
mod regressor_tests;
#[cfg(test)]
mod representation_tests;
#[cfg(test)]
mod stage_tests;
#[cfg(test)]
mod validation_tests;

#[cfg(test)]
mod acceptance_tests;

#[cfg(test)]
mod export_tests;
