//! Pure dispatcher: the fitted SGD regressor model, its hyperparameter
//! accessors, its prediction surface and its incremental learning. Carries no
//! implementation logic.

mod accessors;
mod core_implementation;
mod export;
mod partial_fit;
mod prediction;

pub use core_implementation::SKSGDRegressorModel;
pub use export::{
    SK_SGD_REGRESSOR_COEFFICIENTS_TENSOR, SK_SGD_REGRESSOR_ESTIMATOR_NAME,
    SK_SGD_REGRESSOR_INTERCEPT_TENSOR,
};

#[cfg(test)]
mod export_tests;
