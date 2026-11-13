//! Builders for [`crate::SKBaggingClassifier`] and [`crate::SKBaggingRegressor`].
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier_builder;
mod regressor_builder;

pub use classifier_builder::SKBaggingClassifierBuilder;
pub use regressor_builder::SKBaggingRegressorBuilder;
