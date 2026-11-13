//! Builder-only construction of the grid searches (spec `grid-search`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier_builder;
mod regressor_builder;
mod state;

pub use classifier_builder::SKGridSearchCVClassifierBuilder;
pub use regressor_builder::SKGridSearchCVRegressorBuilder;

#[cfg(test)]
mod builder_tests;
