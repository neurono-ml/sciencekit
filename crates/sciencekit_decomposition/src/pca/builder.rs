//! Mandatory builder for [`SKPCA`] (spec `principal-component-analysis`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, SKFloat, builders::SKBuilder,
    sk_validate_hyperparameter,
};

use super::config::{SKPCAConfiguration, SKPCASolver};
use super::core_implementation::SKPCA;

/// Builder-only construction surface for [`SKPCA`].
#[derive(Debug, Clone, Copy)]
pub struct SKPCABuilder<F: SKFloat> {
    pub(super) state: SKBuilderState,
    pub(super) number_of_components: Option<usize>,
    pub(super) svd_solver: SKPCASolver,
    pub(super) whiten: bool,
    pub(super) tolerance: F,
    pub(super) number_of_power_iterations: usize,
    pub(super) allow_densification: bool,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKPCABuilder<F> {
    /// Creates a builder with the documented defaults: `number_of_components`
    /// unset (`None`, meaning `min(n_samples, n_features)`), `Auto` solver,
    /// `whiten = false`, `tolerance = 0.0`, `5` power iterations,
    /// `allow_densification = false`, `random_seed = 0`, `Faer` backend and
    /// `Automatic` execution mode.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_components: None,
            svd_solver: SKPCASolver::Auto,
            whiten: false,
            tolerance: F::zero(),
            number_of_power_iterations: 5,
            allow_densification: false,
            random_seed: 0,
            backend: SKBackendKind::Faer,
        }
    }

    /// Sets the number of principal components to retain (must be strictly
    /// positive and at most `min(n_samples, n_features)` at fit time).
    pub fn number_of_components(mut self, value: usize) -> Self {
        self.number_of_components = Some(value);
        self
    }

    /// Selects the solver (`Auto`, `Full`, `Randomized` or `CovarianceEigh`).
    pub fn svd_solver(mut self, value: SKPCASolver) -> Self {
        self.svd_solver = value;
        self
    }

    /// Enables or disables unit-variance scaling of transformed components.
    pub fn whiten(mut self, value: bool) -> Self {
        self.whiten = value;
        self
    }

    /// Sets the singular-value tolerance (must be finite and non-negative).
    pub fn tolerance(mut self, value: F) -> Self {
        self.tolerance = value;
        self
    }

    /// Sets the number of power iterations for the randomized SVD solver (must
    /// be strictly positive).
    pub fn number_of_power_iterations(mut self, value: usize) -> Self {
        self.number_of_power_iterations = value;
        self
    }

    /// Explicitly authorizes densifying a sparse CSR view for the dense-only
    /// `Full` and `CovarianceEigh` solvers (with a warning reporting the byte
    /// cost). Defaults to `false` so sparse inputs are never densified
    /// silently.
    pub fn allow_densification(mut self, value: bool) -> Self {
        self.allow_densification = value;
        self
    }

    /// Sets the deterministic seed used by the randomized SVD projection.
    pub fn random_seed(mut self, value: u64) -> Self {
        self.random_seed = value;
        self
    }

    /// Sets the math backend recorded in operation spans and dispatched per
    /// operation.
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

    fn validated_configuration(&self) -> Result<SKPCAConfiguration<F>, SKError> {
        if let Some(components) = self.number_of_components {
            sk_validate_hyperparameter(
                "number_of_components",
                components > 0,
                "must be strictly positive",
            )?;
        }
        sk_validate_hyperparameter(
            "tolerance",
            self.tolerance.is_finite() && self.tolerance >= F::zero(),
            "must be finite and non-negative",
        )?;
        sk_validate_hyperparameter(
            "number_of_power_iterations",
            self.number_of_power_iterations > 0,
            "must be strictly positive",
        )?;
        Ok(SKPCAConfiguration {
            number_of_components: self.number_of_components,
            svd_solver: self.svd_solver,
            whiten: self.whiten,
            tolerance: self.tolerance,
            number_of_power_iterations: self.number_of_power_iterations,
            allow_densification: self.allow_densification,
            random_seed: self.random_seed,
            backend: self.backend,
            builder_state: self.state,
        })
    }
}

impl<F: SKFloat> Default for SKPCABuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKPCA<F>> for SKPCABuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKPCA<F>, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKPCA::from_configuration(configuration))
    }
}
