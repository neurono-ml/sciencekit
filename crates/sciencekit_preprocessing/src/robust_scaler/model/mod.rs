//! Pure dispatcher: the fitted robust scaler model and its online update.
//! Carries no implementation logic.

mod core_implementation;
mod partial_fit;

pub use core_implementation::SKRobustScalerModel;
