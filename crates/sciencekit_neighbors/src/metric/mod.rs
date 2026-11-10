//! The metric catalogue and its geometry bounds (spec
//! `nearest-neighbor-search`).
//!
//! This is a pure dispatcher module: implementation and companion tests live in
//! their own files.

mod catalogue;

pub use catalogue::SKNeighborMetric;

#[cfg(test)]
mod metric_tests;
