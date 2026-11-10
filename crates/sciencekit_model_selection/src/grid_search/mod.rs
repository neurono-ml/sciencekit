//! Hyperparameter grid search with cross-validation (spec `grid-search`).
//!
//! `SKGridSearchCVClassifier` and `SKGridSearchCVRegressor` evaluate a list of
//! fully configured candidates on every split of a configured splitter,
//! aggregate mean and standard deviation per candidate, select the highest mean
//! (ties by candidate index) and optionally refit the winner on all rows.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod classifier;
mod configuration;
mod error_score;
mod evaluation;
mod observation;
mod prediction;
mod regressor;
mod results;
mod scoring;

pub use builder::{SKGridSearchCVClassifierBuilder, SKGridSearchCVRegressorBuilder};
pub use classifier::{SKGridSearchCVClassifier, SKGridSearchCVClassifierModel};
pub use error_score::SKErrorScore;
pub use regressor::{SKGridSearchCVRegressor, SKGridSearchCVRegressorModel};
pub use results::SKGridSearchResults;
pub use sciencekit_metrics::classification::SKAveraging;
pub use scoring::{SKClassifierScoring, SKRegressorScoring, SKScoringFamily};

#[cfg(test)]
pub(crate) mod mock_estimators;
#[cfg(test)]
mod representation_tests;
