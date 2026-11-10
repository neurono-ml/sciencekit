//! The shared searcher contract and its query result types.
//!
//! This is a pure dispatcher module: implementation and companion tests live in
//! their own files.

mod query_result;

pub use query_result::{SKNeighborQuery, SKRadiusNeighbors};

#[cfg(test)]
mod query_result_tests;
