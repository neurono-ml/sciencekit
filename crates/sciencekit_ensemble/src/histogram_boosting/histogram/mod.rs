//! Per-node gradient/Hessian histograms and split gain scans (spec
//! `histogram-gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod accumulate;
mod categorical;
mod gain;
mod subtract;

pub use accumulate::{SKFeatureHistogram, SKNodeHistogram, sk_accumulate_histogram};
pub use categorical::sk_scan_categorical_feature;
pub use gain::{
    SKHistogramSplit, SKHistogramSplitCandidate, sk_scan_numeric_feature, sk_split_gain,
};
pub use subtract::sk_subtract_histogram;

#[cfg(test)]
mod accumulate_tests;
#[cfg(test)]
mod categorical_tests;
#[cfg(test)]
mod gain_tests;
#[cfg(test)]
mod subtract_tests;
