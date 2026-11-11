//! Stage-wise gradient-boosting classifier and regressor (spec
//! `gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod classifier;
mod classifier_builder;
pub(super) mod config;
mod loss_kind;

pub use classifier::SKGradientBoostingClassifier;
pub use classifier_builder::SKGradientBoostingClassifierBuilder;
pub use loss_kind::{SKGradientBoostingClassificationLoss, SKGradientBoostingRegressionLoss};

#[cfg(test)]
mod gb_classifier_builder_tests;
