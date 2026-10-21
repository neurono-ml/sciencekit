//! Pure dispatcher: the fitted linear-regression model and its incremental
//! update. Carries no implementation logic.

mod core_implementation;
mod partial_fit;

pub use core_implementation::SKLinearRegressionModel;
pub(crate) use core_implementation::split_solution;
