//! The typed linear pipeline: builder, finalizers, fitted models and the fit /
//! predict paths.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod classifier;
mod fit;
mod model;
mod predict;
mod regressor;

pub use builder::{SKPipeline, SKPipelineBuilder};
pub use classifier::SKPipelineClassifierBuilder;
pub use model::{SKPipelineClassifier, SKPipelineRegressor};
pub use regressor::SKPipelineRegressorBuilder;

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod fit_tests;
#[cfg(test)]
mod model_tests;
