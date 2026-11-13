//! Pure dispatcher: the fitted robust scaler model and its online update.
//! Carries no implementation logic.

mod core_implementation;
mod export;
mod partial_fit;

pub use core_implementation::SKRobustScalerModel;
pub use export::{
    SK_ROBUST_SCALER_CENTER_TENSOR, SK_ROBUST_SCALER_ESTIMATOR_NAME, SK_ROBUST_SCALER_SCALE_TENSOR,
};
