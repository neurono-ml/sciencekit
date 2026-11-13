//! Pure dispatcher: the fitted SGD classifier model, its hyperparameter
//! accessors, its prediction surface and its incremental learning. Carries no
//! implementation logic.

mod accessors;
mod core_implementation;
mod export;
mod partial_fit;
mod prediction;

pub use core_implementation::SKSGDClassifierModel;
pub use export::{
    SK_SGD_CLASSIFIER_COEFFICIENTS_TENSOR, SK_SGD_CLASSIFIER_ESTIMATOR_NAME,
    SK_SGD_CLASSIFIER_INTERCEPTS_TENSOR,
};
