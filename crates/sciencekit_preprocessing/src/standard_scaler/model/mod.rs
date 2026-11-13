//! Pure dispatcher: the fitted standard-scaler model and its incremental update.
//! Carries no implementation logic.

mod core_implementation;
mod export;
mod partial_fit;

pub use core_implementation::SKStandardScalerModel;
pub(crate) use core_implementation::derive_statistics;
pub use export::{
    SK_STANDARD_SCALER_DEVIATIONS_TENSOR, SK_STANDARD_SCALER_ESTIMATOR_NAME,
    SK_STANDARD_SCALER_MEAN_TENSOR, SK_STANDARD_SCALER_MOMENTS_MEAN_TENSOR,
    SK_STANDARD_SCALER_SCALE_TENSOR,
};

#[cfg(test)]
mod export_tests;
