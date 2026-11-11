//! Configuration shared by the bagging classifier and regressor.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, sk_validate_hyperparameter,
};
use sciencekit_tree::SKMaximumFeatures;

use crate::bootstrap::SKMaximumSamples;

/// Shared hyperparameters for [`crate::SKBaggingClassifier`] and
/// [`crate::SKBaggingRegressor`].
#[derive(Debug, Clone)]
pub(super) struct SKBaggingConfiguration {
    pub(super) state: SKBuilderState,
    pub(super) number_of_estimators: usize,
    pub(super) maximum_samples: SKMaximumSamples<f64>,
    pub(super) maximum_features: SKMaximumFeatures<f64>,
    pub(super) bootstrap: bool,
    pub(super) bootstrap_features: bool,
    pub(super) oob_score: bool,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl SKBaggingConfiguration {
    /// Documented default configuration (spec `bagging-ensembles`).
    pub(super) fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_estimators: 10,
            maximum_samples: SKMaximumSamples::All,
            maximum_features: SKMaximumFeatures::All,
            bootstrap: true,
            bootstrap_features: false,
            oob_score: false,
            random_seed: 0,
            backend: SKBackendKind::Faer,
        }
    }

    /// Configured execution intent.
    pub(super) fn execution_intent(&self) -> SKExecutionMode {
        self.state.execution_intent()
    }

    /// Validate all bagging hyperparameters before building the estimator.
    pub(super) fn validate(&self) -> Result<(), SKError> {
        sk_validate_hyperparameter(
            "number_of_estimators",
            self.number_of_estimators >= 1,
            "number_of_estimators must be at least 1",
        )?;
        match self.maximum_samples {
            SKMaximumSamples::All => {}
            SKMaximumSamples::Count(count) => sk_validate_hyperparameter(
                "maximum_samples",
                count >= 1,
                "an explicit sample count must be at least 1",
            )?,
            SKMaximumSamples::Fraction(fraction) => sk_validate_hyperparameter(
                "maximum_samples",
                fraction.is_finite() && fraction > 0.0 && fraction <= 1.0,
                "a sample fraction must be finite and in (0, 1]",
            )?,
        }
        match self.maximum_features {
            SKMaximumFeatures::Count(count) => sk_validate_hyperparameter(
                "maximum_features",
                count >= 1,
                "an explicit feature count must be at least 1",
            )?,
            SKMaximumFeatures::Fraction(fraction) => sk_validate_hyperparameter(
                "maximum_features",
                fraction.is_finite() && fraction > 0.0 && fraction <= 1.0,
                "a feature fraction must be finite and in (0, 1]",
            )?,
            SKMaximumFeatures::All | SKMaximumFeatures::Sqrt | SKMaximumFeatures::Log2 => {}
        }
        sk_validate_hyperparameter(
            "oob_score",
            !self.oob_score || self.bootstrap,
            "out-of-bag scoring requires bootstrap sampling to be enabled",
        )
    }
}
