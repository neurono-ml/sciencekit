//! The stochastic-gradient linear classifier (spec
//! `stochastic-gradient-classifier`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod accessors;
pub mod builder;
pub mod coefficients;
pub mod core_implementation;
mod fitting;
pub mod model;
mod targets;

pub use builder::SKSGDClassifierBuilder;
pub use coefficients::SKSGDClassifierCoefficients;
pub use core_implementation::SKSGDClassifier;
pub use model::SKSGDClassifierModel;

#[cfg(test)]
mod binary_tests;
#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod early_stopping_tests;
#[cfg(test)]
mod losses_tests;
#[cfg(test)]
mod multiclass_tests;
#[cfg(test)]
mod partial_fit_tests;
#[cfg(test)]
mod streaming_tests;
#[cfg(test)]
mod warm_start_tests;
#[cfg(test)]
mod weighting_tests;
