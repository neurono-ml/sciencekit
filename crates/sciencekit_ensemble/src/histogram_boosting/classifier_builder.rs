//! Builder for [`SKHistGradientBoostingClassifier`].

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFloat};
use sciencekit_tree::SKMaximumFeatures;

use super::classifier::SKHistGradientBoostingClassifier;
use super::config::SKHistogramGradientBoostingConfiguration;

/// Builder for [`SKHistGradientBoostingClassifier`].
#[derive(Debug, Clone)]
pub struct SKHistGradientBoostingClassifierBuilder<F> {
    configuration: SKHistogramGradientBoostingConfiguration<F>,
}

impl<F: SKFloat> SKHistGradientBoostingClassifierBuilder<F> {
    /// Create a builder initialized with the documented histogram defaults.
    pub fn new() -> Self {
        Self {
            configuration: SKHistogramGradientBoostingConfiguration::new(),
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

    /// Configure the maximum number of quantile bins (default `255`).
    pub fn maximum_bins(mut self, maximum_bins: usize) -> Self {
        self.configuration.maximum_bins = maximum_bins;
        self
    }

    /// Configure the maximum number of leaves per tree (default `31`).
    pub fn maximum_leaf_nodes(mut self, maximum_leaf_nodes: usize) -> Self {
        self.configuration.maximum_leaf_nodes = maximum_leaf_nodes;
        self
    }

    /// Configure the minimum samples per leaf (default `20`).
    pub fn minimum_samples_leaf(mut self, minimum_samples_leaf: usize) -> Self {
        self.configuration.minimum_samples_leaf = minimum_samples_leaf;
        self
    }

    /// Configure L2 Hessian regularization `λ >= 0` (default `0.0`).
    pub fn l2_regularization(mut self, l2_regularization: F) -> Self {
        self.configuration.l2_regularization = l2_regularization;
        self
    }

    /// Configure the per-split feature-subsampling policy (default `All`).
    pub fn maximum_features(mut self, maximum_features: SKMaximumFeatures<F>) -> Self {
        self.configuration.maximum_features = maximum_features;
        self
    }

    /// Declare categorical feature columns (default none).
    pub fn categorical_features(mut self, categorical_features: Vec<usize>) -> Self {
        self.configuration.categorical_features = categorical_features;
        self
    }

    /// Configure stochastic row subsampling in `(0, 1]` (default `1.0`).
    pub fn subsample(mut self, subsample: F) -> Self {
        self.configuration.subsample = subsample;
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

    /// Enable early stopping after `n_iter_no_change` non-improving stages.
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

impl<F: SKFloat> Default for SKHistGradientBoostingClassifierBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKHistGradientBoostingClassifier<F>>
    for SKHistGradientBoostingClassifierBuilder<F>
{
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKHistGradientBoostingClassifier<F>, SKError> {
        self.configuration.validate()?;
        Ok(SKHistGradientBoostingClassifier {
            configuration: self.configuration,
        })
    }
}
