//! Truncated Singular Value Decomposition (`SKTruncatedSVD`, spec
//! `truncated-svd`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! its submodules; every implementation lives in its own file.

mod config;
mod model;

pub use config::SKTruncatedSVDAlgorithm;
pub use model::SKTruncatedSVDModel;

#[cfg(test)]
mod truncated_model_tests;
