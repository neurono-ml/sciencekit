//! DBSCAN: density-based spatial clustering of applications with noise
//! (spec `dbscan`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod builder;
mod config;
mod core_implementation;
mod expansion;
mod export;
mod fit;
mod model;
mod predict;
mod region_query;
mod validation;

pub use builder::SKDBSCANBuilder;
pub use config::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};
pub use core_implementation::SKDBSCAN;
pub use model::SKDBSCANModel;

#[cfg(test)]
mod dbscan_builder_tests;
#[cfg(test)]
mod dbscan_fit_tests;
#[cfg(test)]
mod dbscan_model_tests;
#[cfg(test)]
mod dbscan_predict_tests;
#[cfg(test)]
mod dbscan_query_tests;
#[cfg(test)]
mod dbscan_representation_tests;
#[cfg(test)]
mod dbscan_validation_tests;
