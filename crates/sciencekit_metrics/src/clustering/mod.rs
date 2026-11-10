//! Clustering evaluation metrics (spec `clustering-metrics`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod grouping;
mod inputs;
mod silhouette;

pub use silhouette::{sk_silhouette_samples, sk_silhouette_score};

#[cfg(test)]
mod silhouette_tests;
