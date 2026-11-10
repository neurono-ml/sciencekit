//! The mandatory KMeans builder (spec `k-means`, decision 1).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKBuilderState,
    builders::SKBuilder, sk_validate_hyperparameter,
};

use super::SKKMeansAlgorithm;
use super::config::{SKKMeansConfiguration, SKKMeansInitialization};
use super::core_implementation::SKKMeans;

/// Builder-only construction surface for [`SKKMeans`].
pub struct SKKMeansBuilder<F: SKFloat> {
    pub(super) state: SKBuilderState,
    pub(super) number_of_clusters: usize,
    pub(super) initialization: SKKMeansInitialization,
    pub(super) number_of_initializations: Option<usize>,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) tolerance: F,
    pub(super) algorithm: SKKMeansAlgorithm,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKKMeansBuilder<F> {
    /// A builder carrying the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_clusters: 8,
            initialization: SKKMeansInitialization::KMeansPlusPlus,
            number_of_initializations: None,
            maximum_number_of_iterations: 300,
            tolerance: F::from(1e-4).expect("the default tolerance fits the scalar"),
            algorithm: SKKMeansAlgorithm::Auto,
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

    /// Overrides the number of independent initializations.
    ///
    /// When unset, the default follows the initialization strategy: `10` for
    /// random initialization and `1` for k-means++.
    pub fn number_of_initializations(mut self, value: usize) -> Self {
        self.number_of_initializations = Some(value);
        self
    }

    /// Sets the iteration budget (must be strictly positive).
    pub fn maximum_number_of_iterations(mut self, value: usize) -> Self {
        self.maximum_number_of_iterations = value;
        self
    }

    /// Sets the convergence tolerance (must be finite and non-negative).
    pub fn tolerance(mut self, value: F) -> Self {
        self.tolerance = value;
        self
    }

    /// Sets the assignment algorithm.
    pub fn algorithm(mut self, value: SKKMeansAlgorithm) -> Self {
        self.algorithm = value;
        self
    }

    /// Sets the deterministic seed.
    pub fn random_seed(mut self, value: u64) -> Self {
        self.random_seed = value;
        self
    }

    /// Sets the backend kind recorded in operation spans.
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

    /// The effective number of initializations after default resolution.
    fn resolved_number_of_initializations(&self) -> usize {
        self.number_of_initializations
            .unwrap_or(match self.initialization {
                SKKMeansInitialization::Random => 10,
                SKKMeansInitialization::KMeansPlusPlus => 1,
            })
    }

    /// Validates the accumulated configuration.
    fn validated_configuration(&self) -> Result<SKKMeansConfiguration<F>, SKError> {
        sk_validate_hyperparameter(
            "number_of_clusters",
            self.number_of_clusters > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "maximum_number_of_iterations",
            self.maximum_number_of_iterations > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "tolerance",
            self.tolerance.is_finite() && self.tolerance >= F::zero(),
            "must be finite and non-negative",
        )?;
        let number_of_initializations = self.resolved_number_of_initializations();
        sk_validate_hyperparameter(
            "number_of_initializations",
            number_of_initializations > 0,
            "must be strictly positive",
        )?;
        Ok(SKKMeansConfiguration {
            number_of_clusters: self.number_of_clusters,
            initialization: self.initialization,
            number_of_initializations,
            maximum_number_of_iterations: self.maximum_number_of_iterations,
            tolerance: self.tolerance,
            algorithm: self.algorithm,
            random_seed: self.random_seed,
            backend: self.backend,
            execution_mode: self.state.execution_intent(),
        })
    }
}

impl<F: SKFloat> Default for SKKMeansBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKKMeans<F>> for SKKMeansBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKKMeans<F>, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKKMeans::from_configuration(configuration))
    }
}
