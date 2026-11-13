//! Pure dispatcher: the fitted min/max scaler model and its updates. Carries no
//! implementation logic.

mod core_implementation;
mod export;
mod partial_fit;

pub use core_implementation::SKMinMaxScalerModel;
pub use export::{
    SK_MINMAX_SCALER_ESTIMATOR_NAME, SK_MINMAX_SCALER_MAXIMUM_TENSOR,
    SK_MINMAX_SCALER_MINIMUM_TENSOR, SK_MINMAX_SCALER_RANGE_TENSOR, SK_MINMAX_SCALER_SCALE_TENSOR,
};
