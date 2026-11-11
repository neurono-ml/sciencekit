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
pub(super) mod importances;
mod loss_kind;
pub(super) mod validation;

pub use classifier::SKGradientBoostingClassifier;
pub use classifier_builder::SKGradientBoostingClassifierBuilder;
pub use classifier_model::SKGradientBoostingClassifierModel;
pub use loss_kind::{SKGradientBoostingClassificationLoss, SKGradientBoostingRegressionLoss};

#[cfg(test)]
mod gb_classifier_builder_tests;
#[cfg(test)]
mod gb_classifier_tests;
