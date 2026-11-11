//! Feature-subset routing and row-parallel prediction aggregation for bagging.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier_predict;
mod classifier_routing;
mod regressor_predict;
mod regressor_routing;

pub(super) use classifier_predict::sk_select_canonical_bagging_class;
pub(super) use classifier_routing::sk_predict_classifier_member_rows;
#[cfg(test)]
pub(super) use regressor_routing::sk_predict_regressor_chunk;
pub(super) use regressor_routing::sk_predict_regressor_rows;
