//! Principal Component Analysis (`SKPCA`, spec `principal-component-analysis`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! its submodules; every implementation lives in its own file.

mod builder;
mod centering;
mod config;
mod core_implementation;
mod fit;
mod model;
mod solvers;
mod transform;
mod validation;

pub use builder::SKPCABuilder;
pub(crate) use centering::{sk_center_dense, sk_view_dimensions};
pub use config::SKPCASolver;
pub use core_implementation::SKPCA;
pub use model::SKPCAModel;
pub(crate) use solvers::{SKPCADecomposition, sk_compute_variance_ratio, sk_orient_components};
pub(crate) use validation::{sk_validate_finiteness, sk_validate_matrix_input};

#[cfg(test)]
mod pca_builder_tests;
#[cfg(test)]
mod pca_centering_tests;
#[cfg(test)]
mod pca_concurrency_tests;
#[cfg(test)]
mod pca_covariance_tests;
#[cfg(test)]
mod pca_fit_tests;
#[cfg(test)]
mod pca_full_tests;
#[cfg(test)]
mod pca_randomized_tests;
#[cfg(test)]
mod pca_solver_tests;
#[cfg(test)]
mod pca_transform_tests;
#[cfg(test)]
mod pca_validation_tests;
#[cfg(test)]
pub(crate) mod test_tracing;
