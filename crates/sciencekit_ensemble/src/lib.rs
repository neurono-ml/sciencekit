//! `sciencekit_ensemble` — ensemble estimators for sciencekit (specs
//! `random-forests`, `bagging-ensembles` and `gradient-boosting`).
//!
//! The crate ships [`SKRandomForestClassifier`], [`SKRandomForestRegressor`],
//! [`SKBaggingClassifier`], [`SKBaggingRegressor`],
//! [`SKGradientBoostingClassifier`] and [`SKGradientBoostingRegressor`],
//! constructed only through their builders, together with the shared
//! gradient-boosting substrate in [`boosting`].
//!
//! Dense and borrowed CSR designs enter through the shared data boundary and
//! are never densified.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bootstrap;
mod export_support;

pub mod bagging;
pub mod boosting;
pub mod gradient_boosting;
pub mod histogram_boosting;
pub mod random_forest;

pub use bagging::{
    SKBaggingClassifier, SKBaggingClassifierBuilder, SKBaggingClassifierModel, SKBaggingError,
    SKBaggingRegressor, SKBaggingRegressorBuilder, SKBaggingRegressorModel, SKBaseEstimatorError,
};
pub use bootstrap::SKMaximumSamples;
pub use gradient_boosting::{
    SKGradientBoostingClassificationLoss, SKGradientBoostingClassifier,
    SKGradientBoostingClassifierBuilder, SKGradientBoostingClassifierModel,
    SKGradientBoostingRegressionLoss, SKGradientBoostingRegressor,
    SKGradientBoostingRegressorBuilder, SKGradientBoostingRegressorModel,
};
pub use histogram_boosting::{
    SKHistGradientBoostingClassifier, SKHistGradientBoostingClassifierBuilder,
    SKHistGradientBoostingClassifierModel, SKHistGradientBoostingRegressionLoss,
    SKHistGradientBoostingRegressor, SKHistGradientBoostingRegressorBuilder,
    SKHistGradientBoostingRegressorModel,
};
pub use random_forest::{
    SKRandomForestClassifier, SKRandomForestClassifierBuilder, SKRandomForestClassifierModel,
    SKRandomForestRegressor, SKRandomForestRegressorBuilder, SKRandomForestRegressorModel,
};

// Re-export the split vocabulary owned by `sciencekit_tree` so consumers
// configure an ensemble without depending on the tree crate directly.
pub use sciencekit_tree::{SKClassWeight, SKMaximumFeatures, SKTreeCriterion};

#[cfg(test)]
mod acceptance_tests;
