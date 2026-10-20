//! Pure dispatcher: the fitted polynomial-features model and its transforms.
//! Carries no implementation logic.

mod core_implementation;
mod sparse;

pub use core_implementation::SKPolynomialFeaturesModel;
