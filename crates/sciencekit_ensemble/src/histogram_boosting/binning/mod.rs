//! Binned feature representation (spec `histogram-gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod dataset;
mod edges;

pub use dataset::{SKBinnedDataset, SKFeatureBinning};
pub use edges::SKNumericBinning;

#[cfg(test)]
mod binning_tests;
#[cfg(test)]
mod dataset_tests;
