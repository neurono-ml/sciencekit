//! Mandatory builder surface for [`SKMiniBatchKMeans`] (spec
//! `mini-batch-k-means`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, SKFloat, builders::SKBuilder,
    sk_validate_hyperparameter,
};

use super::config::SKMiniBatchKMeansConfiguration;
use super::core_implementation::SKMiniBatchKMeans;
use crate::kmeans::SKKMeansInitialization;

/// Builder-only construction surface for [`SKMiniBatchKMeans`].
pub struct SKMiniBatchKMeansBuilder<F: SKFloat> {
    pub(super) state: SKBuilderState,
    pub(super) number_of_clusters: usize,
    pub(super) initialization: SKKMeansInitialization,
    pub(super) number_of_initializations: usize,
    pub(super) batch_size: usize,
    pub(super) initialization_batch_size: Option<usize>,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) maximum_number_of_epochs: usize,
    pub(super) tolerance: F,
    pub(super) reassignment_ratio: F,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKMiniBatchKMeansBuilder<F> {
    /// Creates a builder carrying the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_clusters: 8,
            initialization: SKKMeansInitialization::KMeansPlusPlus,
            number_of_initializations: 3,
            batch_size: 1024,
            initialization_batch_size: None,
            maximum_number_of_iterations: 100,
            maximum_number_of_epochs: 1,
            tolerance: F::zero(),
            reassignment_ratio: F::from(0.01).expect("default reassignment ratio fits scalar"),
            random_seed: 0,
            backend: SKBackendKind::Faer,
        }
    }

    /// Sets the requested cluster count (must be strictly positive).
    pub fn number_of_clusters(mut self, value: usize) -> Self {
        self.number_of_clusters = value;
        self
    }

    /// Sets the centroid initialization strategy.
    pub fn initialization(mut self, value: SKKMeansInitialization) -> Self {
        self.initialization = value;
        self
    }

    /// Sets the number of independent seeded initializations (must be strictly positive).
    pub fn number_of_initializations(mut self, value: usize) -> Self {
        self.number_of_initializations = value;
        self
    }

    /// Sets the mini-batch size (must be strictly positive).
    pub fn batch_size(mut self, value: usize) -> Self {
        self.batch_size = value;
        self
    }

    /// Overrides the initialization sample bound (default `3 × batch_size`, must be `>= number_of_clusters`).
    pub fn initialization_batch_size(mut self, value: usize) -> Self {
        self.initialization_batch_size = Some(value);
        self
    }

    /// Sets the in-memory pass budget (must be strictly positive).
    pub fn maximum_number_of_iterations(mut self, value: usize) -> Self {
        self.maximum_number_of_iterations = value;
        self
    }

    /// Sets the streaming epoch budget for reopenable providers (must be strictly positive).
    pub fn maximum_number_of_epochs(mut self, value: usize) -> Self {
        self.maximum_number_of_epochs = value;
        self
    }

    /// Sets the convergence tolerance (`0` disables early stopping; must be finite and non-negative).
    pub fn tolerance(mut self, value: F) -> Self {
        self.tolerance = value;
        self
    }

    /// Sets the low-count center reassignment ratio (must be finite and in `[0, 1]`).
    pub fn reassignment_ratio(mut self, value: F) -> Self {
        self.reassignment_ratio = value;
        self
    }

    /// Sets the deterministic seed.
    pub fn random_seed(mut self, value: u64) -> Self {
        self.random_seed = value;
        self
    }

    /// Sets the backend kind recorded in every operation span.
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.backend = value;
        self
    }

    /// Sets the execution intent.
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    /// Validates the accumulated configuration.
    fn validated_configuration(&self) -> Result<SKMiniBatchKMeansConfiguration<F>, SKError> {
        sk_validate_hyperparameter(
            "number_of_clusters",
            self.number_of_clusters > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "number_of_initializations",
            self.number_of_initializations > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "batch_size",
            self.batch_size > 0,
            "must be strictly positive",
        )?;
        let initialization_batch_size = match self.initialization_batch_size {
            Some(explicit) => {
                sk_validate_hyperparameter(
                    "initialization_batch_size",
                    explicit >= self.number_of_clusters,
                    "must be at least number_of_clusters",
                )?;
                explicit
            }
            None => self
                .batch_size
                .saturating_mul(3)
                .max(self.number_of_clusters),
        };
        sk_validate_hyperparameter(
            "maximum_number_of_iterations",
            self.maximum_number_of_iterations > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "maximum_number_of_epochs",
            self.maximum_number_of_epochs > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "tolerance",
            self.tolerance.is_finite() && self.tolerance >= F::zero(),
            "must be finite and non-negative",
        )?;
        sk_validate_hyperparameter(
            "reassignment_ratio",
            self.reassignment_ratio.is_finite()
                && self.reassignment_ratio >= F::zero()
                && self.reassignment_ratio <= F::one(),
            "must be finite and in [0, 1]",
        )?;
        Ok(SKMiniBatchKMeansConfiguration {
            number_of_clusters: self.number_of_clusters,
            initialization: self.initialization,
            number_of_initializations: self.number_of_initializations,
            batch_size: self.batch_size,
            initialization_batch_size,
            maximum_number_of_iterations: self.maximum_number_of_iterations,
            maximum_number_of_epochs: self.maximum_number_of_epochs,
            tolerance: self.tolerance,
            reassignment_ratio: self.reassignment_ratio,
            random_seed: self.random_seed,
            backend: self.backend,
            execution_mode: self.state.execution_intent(),
        })
    }
}

impl<F: SKFloat> Default for SKMiniBatchKMeansBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKMiniBatchKMeans<F>> for SKMiniBatchKMeansBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKMiniBatchKMeans<F>, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKMiniBatchKMeans::from_configuration(configuration))
    }
}
