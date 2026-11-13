//! Pure dispatcher: the fitted lasso model, its accessors and its prediction.
//! Carries no implementation logic.

mod core_implementation;
mod export;
mod predictor;

pub use core_implementation::SKLassoModel;

#[cfg(test)]
mod export_tests;
