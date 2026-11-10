//! Hyperparameter grid search with cross-validation (spec `grid-search`).
//!
//! `SKGridSearchCVClassifier` and `SKGridSearchCVRegressor` evaluate a list of
//! fully configured candidates on every split of a configured splitter,
//! aggregate mean and standard deviation per candidate, select the highest mean
//! (ties by candidate index) and optionally refit the winner on all rows.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod evaluation;
mod scoring;

pub use sciencekit_metrics::classification::SKAveraging;
pub use scoring::{SKClassifierScoring, SKRegressorScoring, SKScoringFamily};

#[cfg(test)]
pub(crate) mod mock_estimators;
