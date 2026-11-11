//! Mandatory builder for [`SKTruncatedSVD`] (spec `truncated-svd`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, SKFloat, builders::SKBuilder,
};

use super::config::{SKTruncatedSVDAlgorithm, SKTruncatedSVDConfiguration};
use super::core_implementation::SKTruncatedSVD;
use super::validation::sk_validate_configuration;

/// Builder-only construction surface for [`SKTruncatedSVD`].
#[derive(Debug, Clone, Copy)]
pub struct SKTruncatedSVDBuilder<F: SKFloat> {
    pub(super) state: SKBuilderState,
    pub(super) number_of_components: usize,
    pub(super) algorithm: SKTruncatedSVDAlgorithm,
    pub(super) number_of_iterations: usize,
    pub(super) tolerance: F,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKTruncatedSVDBuilder<F> {
    /// Creates a builder with the documented defaults: `number_of_components = 2`,
    /// `algorithm = Randomized`, `number_of_iterations = 5`, `tolerance = 0.0`,
    /// `random_seed = 0`, `Faer` backend and `Automatic` execution mode.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_components: 2,
            algorithm: SKTruncatedSVDAlgorithm::Randomized,
            number_of_iterations: 5,
            tolerance: F::zero(),
            random_seed: 0,
            backend: SKBackendKind::Faer,
        }
    }

    /// Sets the number of retained singular components `k` (must lie in
    /// `1..=min(n_samples, n_features)`).
    pub fn number_of_components(mut self, value: usize) -> Self {
        self.number_of_components = value;
        self
    }

    /// Selects the low-rank SVD solver (`Randomized` by default; `Arpack` is
    /// reserved and rejected as an explicit non-goal).
    pub fn algorithm(mut self, value: SKTruncatedSVDAlgorithm) -> Self {
        self.algorithm = value;
        self
    }

    /// Sets the number of power iterations for the randomized range finder
    /// (must be strictly positive).
    pub fn number_of_iterations(mut self, value: usize) -> Self {
        self.number_of_iterations = value;
        self
    }

    /// Sets the singular-value tolerance (must be finite and non-negative).
    pub fn tolerance(mut self, value: F) -> Self {
        self.tolerance = value;
        self
    }

    /// Sets the deterministic seed for the randomized projection.
    pub fn random_seed(mut self, value: u64) -> Self {
        self.random_seed = value;
        self
    }

    /// Sets the math backend recorded in operation spans and dispatched per
    /// operation (`Faer` by default).
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.backend = value;
        self
    }

    /// Sets the execution intent (`Automatic` by default).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn validated_configuration(&self) -> Result<SKTruncatedSVDConfiguration<F>, SKError> {
        sk_validate_configuration(
            self.number_of_components,
            self.algorithm,
            self.number_of_iterations,
            self.tolerance,
        )?;
        Ok(SKTruncatedSVDConfiguration {
            number_of_components: self.number_of_components,
            algorithm: self.algorithm,
            number_of_iterations: self.number_of_iterations,
            tolerance: self.tolerance,
            random_seed: self.random_seed,
            backend: self.backend,
            builder_state: self.state,
        })
    }
}

impl<F: SKFloat> Default for SKTruncatedSVDBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKTruncatedSVD<F>> for SKTruncatedSVDBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKTruncatedSVD<F>, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKTruncatedSVD::from_configuration(configuration))
    }
}
