//! Builder for [`SKGradientBoostingRegressor`]: the sole construction surface.

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFloat, sk_validate_hyperparameter,
};
use sciencekit_tree::SKMaximumFeatures;

use super::config::SKGradientBoostingConfiguration;
use super::loss_kind::SKGradientBoostingRegressionLoss;
use super::regressor::SKGradientBoostingRegressor;

/// Builder for [`SKGradientBoostingRegressor`].
#[derive(Debug, Clone)]
pub struct SKGradientBoostingRegressorBuilder<F> {
    configuration: SKGradientBoostingConfiguration<F>,
    loss: SKGradientBoostingRegressionLoss,
    huber_delta: F,
}

impl<F: SKFloat> SKGradientBoostingRegressorBuilder<F> {
    /// Create a builder initialized with the documented boosting defaults.
    pub fn new() -> Self {
        Self {
            configuration: SKGradientBoostingConfiguration::new(),
            loss: SKGradientBoostingRegressionLoss::SquaredError,
            huber_delta: F::one(),
        }
    }

    /// Configure the maximum number of boosting stages (default `100`).
    pub fn number_of_estimators(mut self, number_of_estimators: usize) -> Self {
        self.configuration.number_of_estimators = number_of_estimators;
        self
    }

    /// Configure the learning-rate shrinkage factor (default `0.1`).
    pub fn learning_rate(mut self, learning_rate: F) -> Self {
        self.configuration.learning_rate = learning_rate;
        self
    }

    /// Configure the regression loss (default `SquaredError`).
    pub fn loss(mut self, loss: SKGradientBoostingRegressionLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Configure the positive Huber clipping threshold `delta` (default `1.0`).
    pub fn huber_delta(mut self, huber_delta: F) -> Self {
        self.huber_delta = huber_delta;
        self
    }

    /// Configure the maximum tree depth per stage (default `3`).
    pub fn maximum_depth(mut self, maximum_depth: usize) -> Self {
        self.configuration.maximum_depth = maximum_depth;
        self
    }

    /// Configure the minimum samples per leaf (default `1`).
    pub fn minimum_samples_leaf(mut self, minimum_samples_leaf: usize) -> Self {
        self.configuration.minimum_samples_leaf = minimum_samples_leaf;
        self
    }

    /// Configure the per-split feature-subsampling policy (default `All`).
    pub fn maximum_features(mut self, maximum_features: SKMaximumFeatures<F>) -> Self {
        self.configuration.maximum_features = maximum_features;
        self
    }

    /// Configure stochastic row subsampling in `(0, 1]` (default `1.0`).
    pub fn subsample(mut self, subsample: F) -> Self {
        self.configuration.subsample = subsample;
        self
    }

    /// Configure L2 Hessian regularization `λ >= 0` (default `0.0`).
    pub fn l2_regularization(mut self, l2_regularization: F) -> Self {
        self.configuration.l2_regularization = l2_regularization;
        self
    }

    /// Configure the deterministic random seed (default `0`).
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.configuration.random_seed = random_seed;
        self
    }

    /// Configure the held-out validation fraction in `(0, 1)` (default `0.1`).
    pub fn validation_fraction(mut self, validation_fraction: F) -> Self {
        self.configuration.validation_fraction = validation_fraction;
        self
    }

    /// Enable early stopping after `n_iter_no_change` non-improving stages
    /// (default unset).
    pub fn n_iter_no_change(mut self, n_iter_no_change: usize) -> Self {
        self.configuration.n_iter_no_change = Some(n_iter_no_change);
        self
    }

    /// Disable early stopping by clearing `n_iter_no_change`.
    pub fn without_early_stopping(mut self) -> Self {
        self.configuration.n_iter_no_change = None;
        self
    }

    /// Configure the early-stopping loss-improvement tolerance (default `1e-4`).
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.configuration.tolerance = tolerance;
        self
    }

    /// Configure the backend recorded in spans and models (default `Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.configuration.backend = backend;
        self
    }

    /// Configure the execution intent (default `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat> Default for SKGradientBoostingRegressorBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKGradientBoostingRegressor<F>>
    for SKGradientBoostingRegressorBuilder<F>
{
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKGradientBoostingRegressor<F>, SKError> {
        self.configuration.validate()?;
        sk_validate_hyperparameter(
            "huber_delta",
            self.huber_delta.is_finite() && self.huber_delta > F::zero(),
            "huber_delta must be positive and finite",
        )?;
        Ok(SKGradientBoostingRegressor {
            configuration: self.configuration,
            loss: self.loss,
            huber_delta: self.huber_delta,
        })
    }
}
