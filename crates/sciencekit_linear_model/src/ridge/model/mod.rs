//! Pure dispatcher: the fitted ridge model, its accessors and its prediction.
//! Carries no implementation logic.

mod core_implementation;
mod export;
mod partial_fit;
mod predictor;

pub use core_implementation::SKRidgeModel;
pub use export::{
    SK_RIDGE_ALPHAS_TENSOR, SK_RIDGE_COEFFICIENTS_TENSOR, SK_RIDGE_ESTIMATOR_NAME,
    SK_RIDGE_INTERCEPTS_TENSOR, SK_RIDGE_SINGULAR_VALUES_TENSOR,
};

#[cfg(test)]
mod export_tests;
