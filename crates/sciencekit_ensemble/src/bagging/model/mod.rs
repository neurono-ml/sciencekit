//! Fitted generic bagging models and out-of-bag state.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier_model;
mod oob_state;
mod regressor_model;

pub(super) use classifier_model::SKBaggingClassifierMember;
pub use classifier_model::SKBaggingClassifierModel;
pub(super) use oob_state::SKBaggingOobState;
pub use regressor_model::SKBaggingRegressorModel;
