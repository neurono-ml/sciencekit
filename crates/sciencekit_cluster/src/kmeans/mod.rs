//! KMeans: seeded k-means++ initialization, Lloyd and Elkan assignment
//! (spec `k-means`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod algorithm;
mod builder;
mod config;
mod core_implementation;
pub(crate) mod distance;
mod elkan;
mod fit;
pub(crate) mod init;
pub(crate) mod iteration;
mod lloyd;
mod model;
mod predict;
mod result;
pub(crate) mod selection;
pub(crate) mod validation;

pub use builder::SKKMeansBuilder;
pub use config::{SKKMeansAlgorithm, SKKMeansInitialization};
pub use core_implementation::SKKMeans;
pub use model::SKKMeansModel;

#[cfg(test)]
mod kmeans_algorithm_tests;
#[cfg(test)]
mod kmeans_builder_tests;
#[cfg(test)]
mod kmeans_distance_tests;
#[cfg(test)]
mod kmeans_elkan_tests;
#[cfg(test)]
mod kmeans_fit_tests;
#[cfg(test)]
mod kmeans_init_tests;
#[cfg(test)]
mod kmeans_lloyd_tests;
#[cfg(test)]
mod kmeans_predict_tests;
#[cfg(test)]
mod kmeans_validation_tests;
