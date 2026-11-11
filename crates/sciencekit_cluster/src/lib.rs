//! `sciencekit_cluster` — unsupervised clustering estimators for sciencekit
//! (PRD §12, Phase 5).
//!
//! The crate provides `SKKMeans` (Lloyd and Elkan k-means with seeded
//! k-means++ initialization) and `SKDBSCAN` (density-based spatial clustering
//! with spatial-tree and sparse brute-force region queries, explicit
//! `SKClusterLabel::Noise` labels and nearest-core prediction). Sparse input
//! is never densified.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dbscan;
pub mod kmeans;

pub use dbscan::{
    SKDBSCAN, SKDBSCANAlgorithm, SKDBSCANBuilder, SKDBSCANMetric, SKDBSCANModel,
    SKDBSCANSearchStrategy,
};
pub use kmeans::{
    SKKMeans, SKKMeansAlgorithm, SKKMeansBuilder, SKKMeansInitialization, SKKMeansModel,
};

#[cfg(test)]
mod acceptance_tests;
