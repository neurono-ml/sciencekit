//! Successive-halving hyperparameter grid search (spec `halving-grid-search`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod resource;
mod scheduler;

pub use resource::SKHalvingResource;
pub use scheduler::{sk_halving_candidate_schedule, sk_halving_resource_schedule};

#[cfg(test)]
mod halving_scheduler_tests;
