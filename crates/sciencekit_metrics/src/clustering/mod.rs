//! Clustering evaluation metrics (spec `clustering-metrics`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod adjusted_rand;
mod calinski_harabasz;
mod contingency;
mod davies_bouldin;
mod geometry;
mod grouping;
mod homogeneity;
mod inputs;
mod mutual_information;
mod silhouette;

pub use adjusted_rand::sk_adjusted_rand_score;
pub use calinski_harabasz::sk_calinski_harabasz_score;
pub use davies_bouldin::sk_davies_bouldin_score;
pub use homogeneity::{sk_completeness_score, sk_homogeneity_score, sk_v_measure_score};
pub use mutual_information::{
    SKMutualInformationNormalization, sk_normalized_mutual_information_score,
};
pub use silhouette::{sk_silhouette_samples, sk_silhouette_score};

#[cfg(test)]
mod adjusted_rand_tests;
#[cfg(test)]
mod calinski_harabasz_tests;
#[cfg(test)]
mod clustering_representation_tests;
#[cfg(test)]
mod davies_bouldin_tests;
#[cfg(test)]
mod homogeneity_tests;
#[cfg(test)]
mod mutual_information_tests;
#[cfg(test)]
mod silhouette_tests;
