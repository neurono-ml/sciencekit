//! DBSCAN: density-based spatial clustering of applications with noise
//! (spec `dbscan`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod config;
mod region_query;

pub use config::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};

#[cfg(test)]
mod dbscan_query_tests;
