//! The metric catalogue and its geometry bounds (spec
//! `nearest-neighbor-search`).
//!
//! This is a pure dispatcher module: implementation and companion tests live in
//! their own files.

mod catalogue;
mod tolerance;

pub use catalogue::{SKNeighborMetric, SKRequestedNeighborMetric};
pub(crate) use tolerance::sk_lower_bound_exceeds;

#[cfg(test)]
mod metric_bounds_tests;
#[cfg(test)]
mod metric_tests;
