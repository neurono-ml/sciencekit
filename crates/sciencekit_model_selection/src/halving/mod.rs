//! Successive-halving hyperparameter grid search (spec `halving-grid-search`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod classifier;
mod configuration;
mod evaluation;
mod regressor;
mod resource;
mod scheduler;
mod subsample;
mod trace;

pub use builder::{SKHalvingGridSearchClassifierBuilder, SKHalvingGridSearchRegressorBuilder};
pub use classifier::{SKHalvingGridSearchClassifier, SKHalvingGridSearchClassifierModel};
pub use configuration::SKHalvingScoring;
pub use regressor::{SKHalvingGridSearchRegressor, SKHalvingGridSearchRegressorModel};
pub use resource::SKHalvingResource;
pub use scheduler::{sk_halving_candidate_schedule, sk_halving_resource_schedule};
pub use trace::SKHalvingRound;

#[cfg(test)]
mod halving_aggressive_tests;
#[cfg(test)]
mod halving_evaluation_tests;
#[cfg(test)]
mod halving_scheduler_tests;
#[cfg(test)]
mod halving_subsample_tests;
#[cfg(test)]
mod halving_trace_tests;
