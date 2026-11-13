//! Shared hyperparameter state and validation for both gradient-boosting
//! estimators.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKFloat, sk_validate_hyperparameter,
};
use sciencekit_tree::SKMaximumFeatures;

use crate::boosting::early_stopping::SKEarlyStoppingConfig;
use crate::boosting::stage::SKBoostingStageParameters;

/// Shared configuration carried by both gradient-boosting builders and
/// estimators.
#[derive(Debug, Clone)]
pub(super) struct SKGradientBoostingConfiguration<F> {
    pub(super) state: SKBuilderState,
    pub(super) number_of_estimators: usize,
    pub(super) learning_rate: F,
    pub(super) maximum_depth: usize,
    pub(super) minimum_samples_leaf: usize,
    pub(super) maximum_features: SKMaximumFeatures<F>,
    pub(super) subsample: F,
    pub(super) l2_regularization: F,
    pub(super) random_seed: u64,
    pub(super) validation_fraction: F,
    pub(super) n_iter_no_change: Option<usize>,
    pub(super) tolerance: F,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKGradientBoostingConfiguration<F> {
    pub(super) fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_estimators: 100,
            learning_rate: F::from(0.1).expect("0.1 fits in scalar"),
            maximum_depth: 3,
            minimum_samples_leaf: 1,
            maximum_features: SKMaximumFeatures::All,
            subsample: F::one(),
            l2_regularization: F::zero(),
            random_seed: 0,
            validation_fraction: F::from(0.1).expect("0.1 fits in scalar"),
            n_iter_no_change: None,
            tolerance: F::from(1e-4).expect("1e-4 fits in scalar"),
            backend: SKBackendKind::Faer,
        }
    }

    pub(super) fn validate(&self) -> Result<(), SKError> {
        sk_validate_hyperparameter(
            "number_of_estimators",
            self.number_of_estimators >= 1,
            "number_of_estimators must be at least 1",
        )?;
        sk_validate_hyperparameter(
            "learning_rate",
            self.learning_rate.is_finite() && self.learning_rate > F::zero(),
            "learning_rate must be positive and finite",
        )?;
        sk_validate_hyperparameter(
            "minimum_samples_leaf",
            self.minimum_samples_leaf >= 1,
            "minimum_samples_leaf must be at least 1",
        )?;
        match &self.maximum_features {
            SKMaximumFeatures::Count(count) => sk_validate_hyperparameter(
                "maximum_features",
                *count >= 1,
                "an explicit feature count must be at least 1",
            )?,
            SKMaximumFeatures::Fraction(fraction) => sk_validate_hyperparameter(
                "maximum_features",
                fraction.is_finite() && *fraction > F::zero() && *fraction <= F::one(),
                "a feature fraction must be finite and in (0, 1]",
            )?,
            _ => {}
        }
        sk_validate_hyperparameter(
            "subsample",
            self.subsample.is_finite() && self.subsample > F::zero() && self.subsample <= F::one(),
            "subsample must be finite and in (0, 1]",
        )?;
        sk_validate_hyperparameter(
            "l2_regularization",
            self.l2_regularization.is_finite() && self.l2_regularization >= F::zero(),
            "l2_regularization must be finite and non-negative",
        )?;
        sk_validate_hyperparameter(
            "validation_fraction",
            self.validation_fraction.is_finite()
                && self.validation_fraction > F::zero()
                && self.validation_fraction < F::one(),
            "validation_fraction must be finite and in (0, 1)",
        )?;
        if let Some(patience) = self.n_iter_no_change {
            sk_validate_hyperparameter(
                "n_iter_no_change",
                patience >= 1,
                "n_iter_no_change must be at least 1 when set",
            )?;
        }
        sk_validate_hyperparameter(
            "tolerance",
            self.tolerance.is_finite() && self.tolerance >= F::zero(),
            "tolerance must be finite and non-negative",
        )
    }

    pub(super) fn stage_parameters(&self) -> SKBoostingStageParameters<F> {
        SKBoostingStageParameters {
            learning_rate: self.learning_rate,
            maximum_depth: self.maximum_depth,
            minimum_samples_leaf: self.minimum_samples_leaf,
            maximum_features: self.maximum_features.clone(),
            subsample: self.subsample,
            l2_regularization: self.l2_regularization,
            random_seed: self.random_seed,
        }
    }

    pub(super) fn early_stopping_config(&self) -> Option<SKEarlyStoppingConfig<F>> {
        self.n_iter_no_change.map(|patience| SKEarlyStoppingConfig {
            validation_fraction: self.validation_fraction,
            patience,
            tolerance: self.tolerance,
            random_seed: self.random_seed,
        })
    }
}
