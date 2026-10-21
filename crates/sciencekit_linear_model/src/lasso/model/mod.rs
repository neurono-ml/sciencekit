//! Pure dispatcher: the fitted lasso model, its accessors and its prediction.
//! Carries no implementation logic.

mod core_implementation;
mod predictor;

pub use core_implementation::SKLassoModel;
