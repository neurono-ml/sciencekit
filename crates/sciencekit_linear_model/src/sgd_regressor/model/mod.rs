//! Pure dispatcher: the fitted SGD regressor model, its hyperparameter
//! accessors, its prediction surface and its incremental learning. Carries no
//! implementation logic.

mod accessors;
mod core_implementation;
mod export;
mod partial_fit;
mod prediction;

pub use core_implementation::SKSGDRegressorModel;

#[cfg(test)]
mod export_tests;
