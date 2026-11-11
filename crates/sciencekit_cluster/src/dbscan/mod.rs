//! DBSCAN: density-based spatial clustering of applications with noise
//! (spec `dbscan`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod config;
mod expansion;
mod region_query;

pub use config::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};

#[cfg(test)]
mod dbscan_fit_tests;
#[cfg(test)]
mod dbscan_query_tests;
