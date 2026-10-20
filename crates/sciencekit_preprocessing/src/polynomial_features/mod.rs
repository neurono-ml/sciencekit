//! `SKPolynomialFeatures`: polynomial and interaction expansion
//! (spec `polynomial-features`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file.

mod builder;
mod core_implementation;
mod model;
mod output;
mod terms;

pub use core_implementation::SKPolynomialFeatures;
pub use model::SKPolynomialFeaturesModel;
pub use output::SKExpandedOutput;

#[cfg(test)]
mod polynomial_features_tests;
