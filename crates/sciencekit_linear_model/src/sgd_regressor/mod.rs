//! The stochastic-gradient linear regressor (spec
//! `stochastic-gradient-regressor`).
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

pub use builder::SKSGDRegressorBuilder;
pub use coefficients::SKSGDRegressorCoefficients;
pub use core_implementation::SKSGDRegressor;
pub use model::SKSGDRegressorModel;

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod dense_tests;
#[cfg(test)]
mod losses_tests;
#[cfg(test)]
mod penalties_tests;
