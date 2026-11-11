//! Builder for [`SKBaggingRegressor`]: the sole construction surface.

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};
use sciencekit_tree::SKMaximumFeatures;

use crate::bootstrap::SKMaximumSamples;

use super::super::config::SKBaggingConfiguration;
use super::super::regressor::SKBaggingRegressor;

/// Builder for [`SKBaggingRegressor`].
pub struct SKBaggingRegressorBuilder<B> {
    base_estimator: B,
    configuration: SKBaggingConfiguration,
}

impl<B> SKBaggingRegressorBuilder<B> {
    /// Create a builder initialized with the documented bagging defaults.
    pub fn new(base_estimator: B) -> Self {
        Self {
            base_estimator,
            configuration: SKBaggingConfiguration::new(),
        }
    }

    /// Replace the base estimator template.
    pub fn base_estimator(mut self, base_estimator: B) -> Self {
        self.base_estimator = base_estimator;
        self
    }

    /// Configure the number of base estimators (default `10`).
    pub fn number_of_estimators(mut self, number_of_estimators: usize) -> Self {
        self.configuration.number_of_estimators = number_of_estimators;
        self
    }

    /// Configure the per-estimator sample-size policy (default `All`).
    pub fn maximum_samples(mut self, maximum_samples: SKMaximumSamples<f64>) -> Self {
        self.configuration.maximum_samples = maximum_samples;
        self
    }

    /// Configure the per-estimator feature-subset policy (default `All`).
    pub fn maximum_features(mut self, maximum_features: SKMaximumFeatures<f64>) -> Self {
        self.configuration.maximum_features = maximum_features;
        self
    }

    /// Configure whether rows are sampled with replacement (default `true`).
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.configuration.bootstrap = bootstrap;
        self
    }

    /// Configure whether features are sampled with replacement (default `false`).
    pub fn bootstrap_features(mut self, bootstrap_features: bool) -> Self {
        self.configuration.bootstrap_features = bootstrap_features;
        self
    }

    /// Configure whether out-of-bag R² is estimated (default `false`).
    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.configuration.oob_score = oob_score;
        self
    }

    /// Configure the global deterministic random seed (default `0`).
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.configuration.random_seed = random_seed;
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

impl<B> SKBuilder<SKBaggingRegressor<B>> for SKBaggingRegressorBuilder<B> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKBaggingRegressor<B>, SKError> {
        self.configuration.validate()?;
        Ok(SKBaggingRegressor {
            base_estimator: self.base_estimator,
            configuration: self.configuration,
        })
    }
}
