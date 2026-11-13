//! Cross-validation scoring (spec `model-selection-splits`, requirement
//! `Cross-validation scoring`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier;
mod fitting;
mod regressor;

pub use classifier::sk_cross_val_score_classifier;
pub use regressor::sk_cross_val_score_regressor;

pub(crate) use fitting::{sk_fit_on_partition, sk_fit_on_views};

#[cfg(test)]
mod cross_validation_tests;
#[cfg(test)]
pub(crate) mod mock_estimators;
