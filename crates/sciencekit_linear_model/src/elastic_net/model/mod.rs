//! Pure dispatcher: the fitted elastic-net model, its accessors and its
//! prediction. Carries no implementation logic.

mod core_implementation;
mod predictor;

pub use core_implementation::SKElasticNetModel;
