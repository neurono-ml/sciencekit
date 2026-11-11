//! Truncated Singular Value Decomposition (`SKTruncatedSVD`, spec
//! `truncated-svd`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! its submodules; every implementation lives in its own file.

mod builder;
mod config;
mod core_implementation;
mod fit;
mod model;
mod product;
mod transform;
mod validation;
mod variance;

pub use builder::SKTruncatedSVDBuilder;
pub use config::{SK_TRUNCATED_SVD_PARITY_TOLERANCE, SKTruncatedSVDAlgorithm};
pub use core_implementation::SKTruncatedSVD;
pub use model::SKTruncatedSVDModel;

#[cfg(test)]
mod truncated_builder_tests;
#[cfg(test)]
mod truncated_fit_tests;
#[cfg(test)]
mod truncated_model_tests;
#[cfg(test)]
mod truncated_representation_tests;
#[cfg(test)]
mod truncated_transform_tests;
#[cfg(test)]
mod truncated_validation_tests;
