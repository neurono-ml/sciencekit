//! `sciencekit_outlier` — anomaly and outlier detection for `sciencekit`
//! (specs `outlier-prediction` and `isolation-forest`).
//!
//! The crate ships [`SKIsolationForest`], constructed exclusively through
//! [`SKIsolationForestBuilder`]. Fitting grows a deterministic ensemble of
//! random isolation trees over seeded subsamples using the shared
//! `sciencekit_tree::SKTree` node layout, without densifying sparse CSR inputs.
//!
//! **Naming:** public items follow PRD §3.4 — structs, enums and traits use
//! `SK`, free-scope functions and constants use `sk_`/`SK_`, and methods carry
//! no prefix.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod isolation_forest;

pub use isolation_forest::{
    SK_ISOLATION_AUTO_OFFSET, SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, SKContamination,
    SKIsolationForest, SKIsolationForestBuilder, SKIsolationForestModel, SKIsolationMaximumSamples,
};
pub use sciencekit_common::{SKOutlierLabel, SKOutlierPredictor};

#[cfg(test)]
mod acceptance_tests;
