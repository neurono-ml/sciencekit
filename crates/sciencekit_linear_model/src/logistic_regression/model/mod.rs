//! Pure dispatcher: the fitted logistic-regression model, its prediction and
//! its incremental learning. Carries no implementation logic.

mod core_implementation;
mod partial_fit;
mod prediction;

pub use core_implementation::SKLogisticRegressionModel;
