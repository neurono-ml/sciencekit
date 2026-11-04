//! The stochastic-gradient configuration and its validation (spec
//! `stochastic-gradient-solver`).

use sciencekit_common::builders::sk_validate_hyperparameter;
use sciencekit_common::{SKError, SKFloat};

use super::learning_rate::SKSGDLearningRate;
use super::penalty::SKSGDPenalty;

/// The validated configuration shared by the classifier and regressor fits.
#[derive(Debug, Clone)]
pub(crate) struct SKSGDConfig<F> {
    /// The penalty.
    pub(crate) penalty: SKSGDPenalty,
    /// The regularization strength.
    pub(crate) alpha: F,
    /// The elastic-net mixing parameter (honored only with `ElasticNet`).
    pub(crate) l1_ratio: F,
    /// Whether the fit solves for an unpenalized intercept.
    pub(crate) include_intercept: bool,
    /// The learning-rate schedule.
    pub(crate) learning_rate: SKSGDLearningRate,
    /// The initial learning rate, consulted by every schedule but `Optimal`.
    pub(crate) initial_learning_rate: F,
    /// The decay power of `InverseScaling`.
    pub(crate) inverse_scaling_power: F,
    /// The convergence tolerance.
    pub(crate) tolerance: F,
    /// The epoch budget of an in-memory fit.
    pub(crate) maximum_number_of_iterations: usize,
    /// The consecutive non-improving epochs tolerated before a plateau.
    pub(crate) number_of_epochs_without_improvement: usize,
    /// Whether a validation slice drives early stopping.
    pub(crate) early_stopping: bool,
    /// The fraction of rows reserved for validation.
    pub(crate) validation_fraction: F,
    /// Whether each epoch shuffles its row order.
    pub(crate) shuffle: bool,
    /// The seed of the shuffler.
    pub(crate) random_seed: u64,
    /// Whether the fit reports the averaged iterate.
    pub(crate) average: bool,
}

impl<F: SKFloat> Default for SKSGDConfig<F> {
    fn default() -> Self {
        Self {
            penalty: SKSGDPenalty::default(),
            alpha: F::from(1e-4).expect("the default alpha is representable"),
            l1_ratio: F::from(0.15).expect("the default l1 ratio is representable"),
            include_intercept: true,
            learning_rate: SKSGDLearningRate::Constant,
            initial_learning_rate: F::from(0.01).expect("the default rate is representable"),
            inverse_scaling_power: F::from(0.5).expect("the default power is representable"),
            tolerance: F::from(1e-3).expect("the default tolerance is representable"),
            maximum_number_of_iterations: 1000,
            number_of_epochs_without_improvement: 5,
            early_stopping: false,
            validation_fraction: F::from(0.1).expect("the default fraction is representable"),
            shuffle: true,
            random_seed: 0,
            average: false,
        }
    }
}

/// Validate the stochastic-gradient configuration.
///
/// Every rejected value is reported as [`SKError::InvalidHyperparameter`]
/// naming the offending field; the combination checks name the selection that
/// makes another field invalid.
pub(crate) fn sk_validate_sgd_configuration<F: SKFloat>(
    config: &SKSGDConfig<F>,
) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "alpha",
        config.alpha >= F::zero() && config.alpha.is_finite(),
        "must be non-negative and finite",
    )?;
    sk_validate_hyperparameter(
        "l1_ratio",
        config.l1_ratio >= F::zero() && config.l1_ratio <= F::one(),
        "must lie in the closed interval [0, 1]",
    )?;
    sk_validate_hyperparameter(
        "tolerance",
        config.tolerance > F::zero(),
        "must be positive",
    )?;
    sk_validate_hyperparameter(
        "maximum_number_of_iterations",
        config.maximum_number_of_iterations > 0,
        "must be at least 1",
    )?;
    sk_validate_hyperparameter(
        "initial_learning_rate",
        !config.learning_rate.uses_initial_rate()
            || (config.initial_learning_rate > F::zero()
                && config.initial_learning_rate.is_finite()),
        "must be positive and finite for the selected schedule",
    )?;
    sk_validate_hyperparameter(
        "inverse_scaling_power",
        config.inverse_scaling_power > F::zero() && config.inverse_scaling_power.is_finite(),
        "must be positive and finite",
    )?;
    sk_validate_hyperparameter(
        "validation_fraction",
        config.validation_fraction > F::zero() && config.validation_fraction < F::one(),
        "must lie in the open interval (0, 1)",
    )?;
    sk_validate_hyperparameter(
        "number_of_epochs_without_improvement",
        config.number_of_epochs_without_improvement > 0,
        "must be at least 1",
    )?;
    sk_validate_hyperparameter(
        "learning_rate",
        config.learning_rate != SKSGDLearningRate::Optimal || config.alpha > F::zero(),
        "the Optimal schedule requires a positive alpha",
    )?;
    sk_validate_hyperparameter(
        "early_stopping",
        !config.early_stopping || config.maximum_number_of_iterations > 1,
        "early stopping requires more than one epoch of budget",
    )
}
