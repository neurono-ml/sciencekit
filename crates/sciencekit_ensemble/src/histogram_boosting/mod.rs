//! Histogram gradient boosting (spec `histogram-gradient-boosting`): quantile
//! binning, histogram split finding, leaf-wise growth, categorical partitions,
//! missing-value routing and the boosted classifier/regressor surface.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

pub mod binning;
pub mod growth;
pub mod histogram;
