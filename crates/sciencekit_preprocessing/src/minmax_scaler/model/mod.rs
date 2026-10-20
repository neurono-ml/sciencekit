//! Pure dispatcher: the fitted min/max scaler model and its updates. Carries no
//! implementation logic.

mod core_implementation;
mod partial_fit;

pub use core_implementation::SKMinMaxScalerModel;
