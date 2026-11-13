//! Stage-wise gradient-boosting classifier and regressor (spec
//! `gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod classifier;
mod classifier_builder;
mod classifier_fit;
mod classifier_model;
mod classifier_predict;
pub(super) mod config;
mod export;
pub(super) mod importances;
mod loss_kind;
mod regressor;
mod regressor_builder;
mod regressor_fit;
mod regressor_model;
mod regressor_predict;
pub(super) mod validation;

pub use classifier::SKGradientBoostingClassifier;
pub use classifier_builder::SKGradientBoostingClassifierBuilder;
pub use classifier_model::SKGradientBoostingClassifierModel;
pub use loss_kind::{SKGradientBoostingClassificationLoss, SKGradientBoostingRegressionLoss};
pub use regressor::SKGradientBoostingRegressor;
pub use regressor_builder::SKGradientBoostingRegressorBuilder;
pub use regressor_model::SKGradientBoostingRegressorModel;

#[cfg(test)]
mod gb_classifier_builder_tests;
#[cfg(test)]
mod gb_classifier_multiclass_tests;
#[cfg(test)]
mod gb_classifier_tests;
#[cfg(test)]
mod gb_determinism_tests;
#[cfg(test)]
mod gb_importances_tests;
#[cfg(test)]
mod gb_regressor_builder_tests;
#[cfg(test)]
mod gb_regressor_tests;
#[cfg(test)]
mod gb_representation_tests;
#[cfg(test)]
mod gb_validation_tests;

#[cfg(test)]
mod export_tests;
