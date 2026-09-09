//! Fit and transformation contracts (spec `estimator-contracts`).
//!
//! Two fit traits separated by the supervision axis. Fit returns a **distinct
//! model type** (sole bearer of the learned state), operates on a **shared
//! reference** of the configured estimator, and produces models that are
//! `Send + Sync` by construction. Prediction exists only on the model type, so
//! predict-without-fit is unrepresentable. The transformer declares its output
//! as an associated type for static pipeline chaining.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod predictor;
mod supervised;
mod transformer;
mod unsupervised;

pub use predictor::SKPredictor;
pub use supervised::SKSupervisedFit;
pub use transformer::SKFeatureTransformer;
pub use unsupervised::SKUnsupervisedFit;

#[cfg(test)]
mod fit_traits_tests;
