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
pub mod kd_tree;
pub mod metric;
pub mod searcher;

pub use searcher::{SKNeighborQuery, SKRadiusNeighbors};

#[cfg(test)]
mod acceptance_tests;
