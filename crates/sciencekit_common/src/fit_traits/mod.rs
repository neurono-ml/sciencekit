//! Fit and transformation contracts (spec `estimator-contracts`).
//!
//! Two fit traits separated by the supervision axis. Fit returns a **distinct
//! model type** (sole bearer of the learned state), operates on a **shared
//! reference** of the configured estimator, and produces models that are
//! `Send + Sync` by construction. Prediction exists only on the model type, so
//! predict-without-fit is unrepresentable: regressors predict responses in the
//! model scalar (`SKRegressorPredictor`), classifiers predict scalar-independent
//! labels plus scalar-preserving probabilities (`SKClassifierPredictor`). The
//! transformer declares its output as an associated type for static pipeline
//! chaining. Clustering exposes typed assignments through
//! [`SKClustererPredictor`], where a label is a cluster identifier or explicit
//! noise — never a numeric sentinel.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod clusterer;
mod outlier;
mod predictor;
mod supervised;
mod transformer;
mod unsupervised;

pub use clusterer::{
    SKClusterIdentifier, SKClusterLabel, SKClustererPredictor, sk_validate_cluster_assignments,
};
pub use outlier::{SKOutlierLabel, SKOutlierPredictor};
pub use predictor::{SKClassifierPredictor, SKRegressorPredictor};
pub use supervised::SKSupervisedFit;
pub use transformer::SKFeatureTransformer;
pub use unsupervised::SKUnsupervisedFit;

#[cfg(test)]
mod clusterer_tests;
#[cfg(test)]
mod fit_traits_tests;
#[cfg(test)]
mod outlier_tests;
