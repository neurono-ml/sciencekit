//! Deterministic estimators used by the grid-search TDD suites.
//!
//! They are intentionally tiny: a constant regressor whose response is the
//! configured hyperparameter, a threshold classifier whose decision boundary is
//! the configured hyperparameter, a variant that rejects dense features (to
//! prove the grid never densifies) and fixed-output models for hand-computed
//! scoring checks.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifiers;
mod fixed_models;
mod helpers;
mod regressors;

pub(crate) use classifiers::SKMockThresholdClassifier;
pub(crate) use fixed_models::{SKMockFixedClassifierModel, SKMockFixedRegressorModel};
pub(crate) use regressors::{SKMockConstantRegressor, SKMockSparseOnlyRegressor};
