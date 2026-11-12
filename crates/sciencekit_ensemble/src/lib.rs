//! `sciencekit_ensemble` — ensemble estimators for sciencekit (specs
//! `random-forests` and `bagging-ensembles`).
//!
//! The crate ships [`SKRandomForestClassifier`], [`SKRandomForestRegressor`],
//! [`SKBaggingClassifier`] and [`SKBaggingRegressor`], constructed only through
//! their builders. Random forests grow one CART tree per bootstrap sample via
//! [`sciencekit_tree`], while generic bagging composes any compatible base
//! estimator over seeded row and feature subsets with strict column routing.
//!
//! Each tree derives its own seed from the forest seed and its index, so a
//! forest is identical across repeated fits and across thread counts. Dense and
//! borrowed CSR designs enter through the shared data boundary and are never
//! densified.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bootstrap;

pub mod bagging;
pub mod random_forest;

pub use bagging::{
    SKBaggingClassifier, SKBaggingClassifierBuilder, SKBaggingClassifierModel, SKBaggingError,
    SKBaggingRegressor, SKBaggingRegressorBuilder, SKBaggingRegressorModel, SKBaseEstimatorError,
};
pub use bootstrap::SKMaximumSamples;
pub use random_forest::{
    SKRandomForestClassifier, SKRandomForestClassifierBuilder, SKRandomForestClassifierModel,
    SKRandomForestRegressor, SKRandomForestRegressorBuilder, SKRandomForestRegressorModel,
};

// Re-export the split vocabulary owned by `sciencekit_tree` so consumers
// configure a forest without depending on the tree crate directly.
pub use sciencekit_tree::{SKClassWeight, SKMaximumFeatures, SKTreeCriterion};

#[cfg(test)]
mod acceptance_tests;
