//! `sciencekit_tree` — CART decision trees for classification and regression
//! (spec `decision-trees`).
//!
//! The crate ships [`SKDecisionTreeClassifier`] and [`SKDecisionTreeRegressor`],
//! both constructed only through their builders. Fitting grows a binary tree by
//! recursive partitioning: at each node the split that maximizes the impurity
//! decrease is chosen, subject to the configured depth, sample-count and
//! feature-subsampling constraints.
//!
//! Dense and borrowed CSR designs enter through the shared data boundary and
//! are dispatched once per operation. A sparse design is re-indexed into a CSC
//! copy at fit time and scanned column-wise; it is never densified. Prediction
//! reads a sparse row's stored entries and treats an absent entry as the
//! observed zero.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod class_weight;
mod criterion;
mod design;
mod growth;
mod splitter;

pub mod impurity;
pub mod node;

pub use class_weight::SKClassWeight;

pub mod classifier;
pub mod regressor;

pub use criterion::SKTreeCriterion;
pub use node::{SKTree, SKTreeNode};
pub use regressor::{
    SKDecisionTreeRegressor, SKDecisionTreeRegressorBuilder, SKDecisionTreeRegressorModel,
};
pub use splitter::{SKMaximumFeatures, SKSplitCandidate, SKSplitter};

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod importances_tests;
#[cfg(test)]
mod representation_tests;
