//! Pure dispatcher: the fitted SGD classifier model, its hyperparameter
//! accessors, its prediction surface and its incremental learning. Carries no
//! implementation logic.

mod accessors;
mod core_implementation;
mod export;
mod partial_fit;
mod prediction;

pub use core_implementation::SKSGDClassifierModel;
