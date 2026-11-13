//! Scoring policies for the grid searches (spec `grid-search`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

use sciencekit_common::SKError;

mod classifier;
mod regressor;

pub use classifier::SKClassifierScoring;
pub use regressor::SKRegressorScoring;

/// The estimator family a scoring policy belongs to.
///
/// The builder validates the configured policy against the family it is
/// building, so a regressor policy paired with a classifier estimator is
/// rejected at construction time instead of failing mid-fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKScoringFamily {
    /// Classification metrics (accuracy, F1, AUC, ...).
    Classifier,
    /// Regression metrics (negative MSE/MAE, R²).
    Regressor,
}

impl SKScoringFamily {
    /// The diagnostic name of the family.
    pub(crate) fn name(self) -> &'static str {
        match self {
            SKScoringFamily::Classifier => "classifier",
            SKScoringFamily::Regressor => "regressor",
        }
    }
}

/// Rejects a scoring policy that does not belong to the estimator family.
pub(crate) fn sk_validate_scoring_family(
    found: SKScoringFamily,
    expected: SKScoringFamily,
) -> Result<(), SKError> {
    if found == expected {
        return Ok(());
    }
    Err(SKError::InvalidHyperparameter {
        name: "scoring",
        reason: format!(
            "a {} scoring policy cannot drive a {} grid search",
            found.name(),
            expected.name()
        ),
    })
}

#[cfg(test)]
mod scoring_tests;
