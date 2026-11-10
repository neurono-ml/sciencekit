//! The k-nearest-neighbour classifier (spec `nearest-neighbor-estimators`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod builder;
mod core_implementation;
mod fit;
mod model;
mod predict;
mod targets;
mod voting;

pub use builder::SKKNeighborsClassifierBuilder;
pub use core_implementation::SKKNeighborsClassifier;
pub use model::SKKNeighborsClassifierModel;

#[cfg(test)]
mod classifier_builder_tests;
#[cfg(test)]
mod classifier_fit_tests;
#[cfg(test)]
mod classifier_predict_tests;
#[cfg(test)]
mod classifier_probability_tests;
#[cfg(test)]
mod classifier_target_tests;
