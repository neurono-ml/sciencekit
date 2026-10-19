//! Pure dispatcher: the fitted standard-scaler model and its incremental update.
//! Carries no implementation logic.

mod core_implementation;
mod partial_fit;

pub use core_implementation::SKStandardScalerModel;
pub(crate) use core_implementation::derive_statistics;
