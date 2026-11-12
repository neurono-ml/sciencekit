//! Per-node gradient/Hessian histograms and split gain scans (spec
//! `histogram-gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod accumulate;

pub use accumulate::{SKFeatureHistogram, SKNodeHistogram, sk_accumulate_histogram};

#[cfg(test)]
mod accumulate_tests;
