//! Pure dispatcher: the fitted linear-regression model and its incremental
//! update. Carries no implementation logic.

mod core_implementation;
mod export;
mod partial_fit;

pub use core_implementation::SKLinearRegressionModel;
pub(crate) use core_implementation::split_solution;
pub use export::{
    SK_LINEAR_COEFFICIENTS_TENSOR, SK_LINEAR_INTERCEPT_TENSOR, SK_LINEAR_REGRESSION_ESTIMATOR_NAME,
    SK_LINEAR_REGRESSION_FORMAT_VERSION,
};

#[cfg(test)]
#[path = "export_tests.rs"]
mod export_tests;
