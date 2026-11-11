//! Truncated Singular Value Decomposition (`SKTruncatedSVD`, spec
//! `truncated-svd`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! its submodules; every implementation lives in its own file.

mod config;
mod fit;
mod model;
mod product;
mod validation;
mod variance;

pub use config::{SK_TRUNCATED_SVD_PARITY_TOLERANCE, SKTruncatedSVDAlgorithm};
pub use model::SKTruncatedSVDModel;

#[cfg(test)]
mod truncated_fit_tests;
#[cfg(test)]
mod truncated_model_tests;
#[cfg(test)]
mod truncated_representation_tests;
