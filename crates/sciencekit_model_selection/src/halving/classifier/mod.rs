//! Pure dispatcher for the classifier halving search module.

mod estimator;
mod fit;
mod model;
mod predict;

pub use estimator::SKHalvingGridSearchClassifier;
pub use model::SKHalvingGridSearchClassifierModel;

#[cfg(test)]
mod halving_refit_tests;
