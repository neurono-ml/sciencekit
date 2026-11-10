//! `sciencekit_neighbors` — exact spatial indexes for nearest-neighbor search
//! (spec `nearest-neighbor-search`).
//!
//! The crate ships two immutable dense indexes, `SKKDTree` and `SKBallTree`,
//! sharing the `SKNearestNeighborsSearcher` contract. Both support exact
//! `k`-nearest and radius queries under the `SKNeighborMetric` catalogue
//! (Euclidean, squared Euclidean, Manhattan and Chebyshev), with deterministic
//! ordering: distances ascending and ties broken by the smallest training row
//! index.
//!
//! Tree geometry is dense by construction, so the indexes are the documented
//! dense-only exception: a sparse training or query view is rejected up front
//! with [`sciencekit_common::SKError::UnsupportedRepresentation`] and a
//! suggestion to use the brute-force searcher, and is never densified.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ball_tree;
pub mod classifier;
pub mod kd_tree;
pub mod metric;
pub mod neighbors;
pub mod searcher;

pub use ball_tree::{SKBallTree, SKBallTreeBuilder};
pub use classifier::{
    SKKNeighborsClassifier, SKKNeighborsClassifierBuilder, SKKNeighborsClassifierModel,
};
pub use kd_tree::{SKKDTree, SKKDTreeBuilder};
pub use metric::{SKNeighborMetric, SKRequestedNeighborMetric};
pub use neighbors::{
    SKBruteForceSearcher, SKFeatureRepresentation, SKNeighborsAlgorithm, SKNeighborsMetric,
    SKNeighborsWeights, SKSearchStrategy, SKTrainingData, sk_resolve_search_strategy,
    sk_weighted_average,
};
pub use searcher::{SKNearestNeighborsSearcher, SKNeighborQuery, SKRadiusNeighbors};

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod representation_tests;
#[cfg(test)]
mod validation_tests;
