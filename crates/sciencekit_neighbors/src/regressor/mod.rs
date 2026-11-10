//! The k-nearest-neighbour regressor (spec `nearest-neighbor-estimators`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod builder;
mod core_implementation;

pub use builder::SKKNeighborsRegressorBuilder;
pub use core_implementation::SKKNeighborsRegressor;

#[cfg(test)]
mod regressor_builder_tests;
