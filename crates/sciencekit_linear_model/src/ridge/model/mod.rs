//! Pure dispatcher: the fitted ridge model, its accessors and its prediction.
//! Carries no implementation logic.

mod core_implementation;
mod partial_fit;
mod predictor;

pub use core_implementation::SKRidgeModel;
