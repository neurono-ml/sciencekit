//! Mandatory builder for [`SKIsolationForest`].

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
    sk_validate_hyperparameter,
};

use super::config::{SKContamination, SKIsolationMaximumSamples};
use super::core_implementation::SKIsolationForest;

/// Builder-only construction surface for [`SKIsolationForest`].
#[derive(Debug, Clone, PartialEq)]
pub struct SKIsolationForestBuilder<F: SKFloat> {
    state: SKBuilderState,
    number_of_trees: usize,
    maximum_samples: SKIsolationMaximumSamples<F>,
    contamination: SKContamination<F>,
    maximum_features: F,
    bootstrap: bool,
    random_seed: u64,
    backend: SKBackendKind,
}

impl<F: SKFloat> SKIsolationForestBuilder<F> {
    /// Creates a builder initialized with the documented Isolation Forest defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_trees: 100,
            maximum_samples: SKIsolationMaximumSamples::Auto,
            contamination: SKContamination::Auto,
            maximum_features: F::one(),
            bootstrap: false,
            random_seed: 0,
            backend: SKBackendKind::Faer,
        }
    }

    /// Sets the number of isolation trees (default `100`, must be `>= 1`).
    pub fn number_of_trees(mut self, value: usize) -> Self {
        self.number_of_trees = value;
        self
    }

    /// Sets the subsample size policy per tree (default [`SKIsolationMaximumSamples::Auto`]).
    pub fn maximum_samples(mut self, value: SKIsolationMaximumSamples<F>) -> Self {
        self.maximum_samples = value;
        self
    }

    /// Sets the contamination policy (default [`SKContamination::Auto`], or fraction in `(0, 0.5]`).
    pub fn contamination(mut self, value: SKContamination<F>) -> Self {
        self.contamination = value;
        self
    }

    /// Sets the fraction of features drawn per tree in `(0, 1]` (default `1.0`).
    pub fn maximum_features(mut self, value: F) -> Self {
        self.maximum_features = value;
        self
    }

    /// Sets whether subsamples are drawn with replacement (default `false`).
    pub fn bootstrap(mut self, value: bool) -> Self {
        self.bootstrap = value;
        self
    }

    /// Sets the deterministic random seed (default `0`).
    pub fn random_seed(mut self, value: u64) -> Self {
        self.random_seed = value;
        self
    }

    /// Sets the compute backend recorded in spans (default [`SKBackendKind::Faer`]).
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.backend = value;
        self
    }

    /// Sets the execution intent (default [`SKExecutionMode::Automatic`]).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    /// Validates and builds the configured [`SKIsolationForest`].
    pub fn build(self) -> Result<SKIsolationForest<F>, SKError> {
        <Self as SKBuilder<SKIsolationForest<F>>>::build(self)
    }

    fn validate(&self) -> Result<(), SKError> {
        sk_validate_hyperparameter(
            "number_of_trees",
            self.number_of_trees > 0,
            "must be strictly positive",
        )?;
        match self.maximum_samples {
            SKIsolationMaximumSamples::Auto => {}
            SKIsolationMaximumSamples::Count(count) => {
                sk_validate_hyperparameter(
                    "maximum_samples",
                    count > 0,
                    "explicit sample count must be at least 1",
                )?;
            }
            SKIsolationMaximumSamples::Fraction(fraction) => {
                sk_validate_hyperparameter(
                    "maximum_samples",
                    fraction.is_finite() && fraction > F::zero() && fraction <= F::one(),
                    "sample fraction must be finite and in (0, 1]",
                )?;
            }
        }
        if let SKContamination::Fraction(fraction) = self.contamination {
            let half = F::from(0.5).expect("0.5 fits scalar");
            sk_validate_hyperparameter(
                "contamination",
                fraction.is_finite() && fraction > F::zero() && fraction <= half,
                "contamination fraction must be finite and in (0, 0.5]",
            )?;
        }
        sk_validate_hyperparameter(
            "maximum_features",
            self.maximum_features.is_finite()
                && self.maximum_features > F::zero()
                && self.maximum_features <= F::one(),
            "maximum_features must be finite and in (0, 1]",
        )
    }
}

impl<F: SKFloat> Default for SKIsolationForestBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKIsolationForest<F>> for SKIsolationForestBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKIsolationForest<F>, SKError> {
        self.validate()?;
        Ok(SKIsolationForest {
            state: self.state,
            number_of_trees: self.number_of_trees,
            maximum_samples: self.maximum_samples,
            contamination: self.contamination,
            maximum_features: self.maximum_features,
            bootstrap: self.bootstrap,
            random_seed: self.random_seed,
            backend: self.backend,
        })
    }
}
