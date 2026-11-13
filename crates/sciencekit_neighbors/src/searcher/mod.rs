//! The shared searcher contract and its query result types.
//!
//! This is a pure dispatcher module: implementation and companion tests live in
//! their own files.

pub(crate) mod candidates;
mod contract;
mod query_result;
mod validation;

pub use contract::SKNearestNeighborsSearcher;
pub use query_result::{SKNeighborQuery, SKRadiusNeighbors};
pub(crate) use validation::{sk_require_dense, sk_validate_query, sk_validate_radius};

#[cfg(test)]
mod query_result_tests;
#[cfg(test)]
mod searcher_tests;
#[cfg(test)]
mod validation_tests;
