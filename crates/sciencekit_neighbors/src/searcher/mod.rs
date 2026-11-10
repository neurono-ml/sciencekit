//! The shared searcher contract and its query result types.
//!
//! This is a pure dispatcher module: implementation and companion tests live in
//! their own files.

mod contract;
mod query_result;
mod validation;

pub use contract::SKNearestNeighborsSearcher;
pub use query_result::{SKNeighborQuery, SKRadiusNeighbors};
pub use validation::sk_validate_query;

#[cfg(test)]
mod query_result_tests;
#[cfg(test)]
mod searcher_tests;
#[cfg(test)]
mod validation_tests;
