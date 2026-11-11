//! Parallel and sequential fitting for generic bagging estimators.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier_fit;
mod classifier_slot;
mod execution;
mod regressor_fit;

pub(super) use classifier_fit::sk_fit_bagging_classifier;
#[allow(unused_imports)]
pub(super) use execution::{sk_execute_estimator_slots, sk_map_row_chunks};
pub(super) use regressor_fit::sk_fit_bagging_regressor;
