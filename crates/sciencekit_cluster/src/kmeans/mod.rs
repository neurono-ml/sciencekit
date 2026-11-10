//! KMeans: seeded k-means++ initialization, Lloyd and Elkan assignment
//! (spec `k-means`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod algorithm;
mod config;
mod distance;
mod elkan;
mod init;
mod iteration;
mod lloyd;
mod result;

pub use config::{SKKMeansAlgorithm, SKKMeansInitialization};

pub(crate) use distance::{
    sk_dense_squared_distances_to_centers, sk_squared_distances_to_center,
    sk_squared_distances_to_centers, sk_view_dimensions,
};
pub(crate) use init::{
    sk_choose_lowest_inertia_index, sk_initialization_seed, sk_initialize_centers,
    sk_run_initializations,
};

#[cfg(test)]
mod kmeans_algorithm_tests;
#[cfg(test)]
mod kmeans_distance_tests;
#[cfg(test)]
mod kmeans_elkan_tests;
#[cfg(test)]
mod kmeans_init_tests;
#[cfg(test)]
mod kmeans_lloyd_tests;
