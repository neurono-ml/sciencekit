//! Shared hyperparameter state and validation for both histogram boosters.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKFloat, sk_validate_hyperparameter,
};
use sciencekit_tree::SKMaximumFeatures;

use crate::boosting::early_stopping::SKEarlyStoppingConfig;
use crate::histogram_boosting::growth::SKHistogramGrowthParameters;
use crate::histogram_boosting::stage::SKHistogramStageParameters;

/// Shared configuration carried by both histogram-boosting estimators.
#[derive(Debug, Clone)]
pub(super) struct SKHistogramGradientBoostingConfiguration<F> {
    pub(super) state: SKBuilderState,
    pub(super) number_of_estimators: usize,
    pub(super) learning_rate: F,
    pub(super) maximum_bins: usize,
    pub(super) maximum_leaf_nodes: usize,
    pub(super) minimum_samples_leaf: usize,
    pub(super) l2_regularization: F,
    pub(super) maximum_features: SKMaximumFeatures<F>,
    pub(super) categorical_features: Vec<usize>,
    pub(super) subsample: F,
    pub(super) random_seed: u64,
    pub(super) validation_fraction: F,
    pub(super) n_iter_no_change: Option<usize>,
    pub(super) tolerance: F,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKHistogramGradientBoostingConfiguration<F> {
    pub(super) fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_estimators: 100,
            learning_rate: F::from(0.1).expect("0.1 fits in scalar"),
            maximum_bins: 255,
            maximum_leaf_nodes: 31,
            minimum_samples_leaf: 20,
            l2_regularization: F::zero(),
            maximum_features: SKMaximumFeatures::All,
            categorical_features: Vec::new(),
            subsample: F::one(),
            random_seed: 0,
            validation_fraction: F::from(0.1).expect("0.1 fits in scalar"),
            n_iter_no_change: Some(10),
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
            "maximum_bins",
            (1..=255).contains(&self.maximum_bins),
            "maximum_bins must be in 1..=255",
        )?;
        sk_validate_hyperparameter(
            "maximum_leaf_nodes",
            self.maximum_leaf_nodes >= 2,
            "maximum_leaf_nodes must be at least 2",
        )?;
        sk_validate_hyperparameter(
            "minimum_samples_leaf",
            self.minimum_samples_leaf >= 1,
            "minimum_samples_leaf must be at least 1",
        )?;
        sk_validate_hyperparameter(
            "l2_regularization",
            self.l2_regularization.is_finite() && self.l2_regularization >= F::zero(),
            "l2_regularization must be finite and non-negative",
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

    pub(super) fn stage_parameters(&self) -> SKHistogramStageParameters<F> {
        SKHistogramStageParameters {
            growth: SKHistogramGrowthParameters {
                maximum_leaf_nodes: self.maximum_leaf_nodes,
                minimum_samples_leaf: self.minimum_samples_leaf,
                l2_regularization: self.l2_regularization,
                maximum_features: self.maximum_features.clone(),
                random_seed: self.random_seed,
            },
            learning_rate: self.learning_rate,
            subsample: self.subsample,
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
