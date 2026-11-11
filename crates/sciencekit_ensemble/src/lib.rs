//! `sciencekit_ensemble` — ensemble estimators for sciencekit (spec
//! `random-forests`).
//!
//! The crate ships [`SKRandomForestClassifier`] and [`SKRandomForestRegressor`],
//! constructed only through their builders. A forest grows one CART tree per
//! bootstrap sample, delegating split search to [`sciencekit_tree`], and
//! aggregates the trees into a deterministic prediction.
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

pub mod random_forest;

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
