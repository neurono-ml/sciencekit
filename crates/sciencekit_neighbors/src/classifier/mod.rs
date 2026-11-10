//! The k-nearest-neighbour classifier (spec `nearest-neighbor-estimators`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod builder;
mod core_implementation;

pub use builder::SKKNeighborsClassifierBuilder;
pub use core_implementation::SKKNeighborsClassifier;

#[cfg(test)]
mod classifier_builder_tests;
