//! Pure dispatcher: the fitted logistic-regression model, its prediction and
//! its incremental learning. Carries no implementation logic.

mod core_implementation;
mod export;
mod partial_fit;
mod prediction;

pub use core_implementation::SKLogisticRegressionModel;
pub use export::{
    SK_LOGISTIC_COEFFICIENTS_TENSOR, SK_LOGISTIC_INTERCEPTS_TENSOR,
    SK_LOGISTIC_REGRESSION_ESTIMATOR_NAME,
};

#[cfg(test)]
mod export_tests;
