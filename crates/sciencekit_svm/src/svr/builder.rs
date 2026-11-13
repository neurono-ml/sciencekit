//! Builder-only construction for [`SKSVR`] (spec `support-vector-machines`).

use sciencekit_common::builders::sk_validate_hyperparameter;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
};
use sciencekit_math::{SKKernel, SKKernelGamma, sk_validate_kernel_hyperparameters};

use super::core_implementation::{SKSVR, SKSVRConfiguration};

/// Builder for [`SKSVR`].
pub struct SKSVRBuilder<F: SKFloat> {
    configuration: SKSVRConfiguration<F>,
}

impl<F: SKFloat> SKSVRBuilder<F> {
    /// Create a builder carrying the documented defaults.
    pub fn new() -> Self {
        Self {
            configuration: SKSVRConfiguration {
                regularization_c: F::one(),
                epsilon: F::from(0.1).expect("0.1 fits SKFloat"),
                kernel: SKKernel::RadialBasisFunction,
                degree: 3,
                gamma: SKKernelGamma::Scale,
                coef0: F::zero(),
                tolerance: F::from(1e-3).expect("1e-3 fits SKFloat"),
                maximum_number_of_iterations: 10_000,
                cache_size: 200,
                shrinking: true,
                backend: SKBackendKind::Faer,
                builder_state: SKBuilderState::new(),
            },
        }
    }

    /// Set the regularization parameter `C` (default `1.0`).
    #[allow(non_snake_case)]
    pub fn C(mut self, value: F) -> Self {
        self.configuration.regularization_c = value;
        self
    }

    /// Set the epsilon-insensitive tube half-width (default `0.1`).
    pub fn epsilon(mut self, value: F) -> Self {
        self.configuration.epsilon = value;
        self
    }

    /// Set the kernel function (default `RadialBasisFunction`).
    pub fn kernel(mut self, value: SKKernel) -> Self {
        self.configuration.kernel = value;
        self
    }

    /// Set the polynomial degree (default `3`).
    pub fn degree(mut self, value: u32) -> Self {
        self.configuration.degree = value;
        self
    }

    /// Set the `gamma` policy (default `Scale`).
    pub fn gamma(mut self, value: SKKernelGamma<F>) -> Self {
        self.configuration.gamma = value;
        self
    }

    /// Set the kernel offset `coef0` (default `0.0`).
    pub fn coef0(mut self, value: F) -> Self {
        self.configuration.coef0 = value;
        self
    }

    /// Set the KKT convergence tolerance (default `1e-3`).
    pub fn tolerance(mut self, value: F) -> Self {
        self.configuration.tolerance = value;
        self
    }

    /// Set the maximum number of SMO iterations (default `10_000`).
    pub fn maximum_number_of_iterations(mut self, value: usize) -> Self {
        self.configuration.maximum_number_of_iterations = value;
        self
    }

    /// Set the LRU kernel-row cache budget in MiB (default `200`).
    pub fn cache_size(mut self, value: usize) -> Self {
        self.configuration.cache_size = value;
        self
    }

    /// Enable or disable active-set shrinking (default `true`).
    pub fn shrinking(mut self, value: bool) -> Self {
        self.configuration.shrinking = value;
        self
    }

    /// Set the math backend kind (default `Faer`).
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.configuration.backend = value;
        self
    }

    /// Set the execution mode intent (default `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.builder_state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat> Default for SKSVRBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKSVR<F>> for SKSVRBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.builder_state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKSVR<F>, SKError> {
        let config = &self.configuration;
        sk_validate_hyperparameter(
            "C",
            config.regularization_c > F::zero() && config.regularization_c.is_finite(),
            "regularization strength C must be positive and finite",
        )?;
        sk_validate_hyperparameter(
            "epsilon",
            config.epsilon >= F::zero() && config.epsilon.is_finite(),
            "epsilon must be finite and non-negative",
        )?;
        sk_validate_hyperparameter(
            "tolerance",
            config.tolerance > F::zero() && config.tolerance.is_finite(),
            "tolerance must be positive and finite",
        )?;
        sk_validate_hyperparameter(
            "maximum_number_of_iterations",
            config.maximum_number_of_iterations > 0,
            "maximum_number_of_iterations must be at least 1",
        )?;
        sk_validate_kernel_hyperparameters(
            config.kernel,
            config.gamma,
            config.degree,
            config.coef0,
        )?;
        Ok(SKSVR::from_configuration(self.configuration))
    }
}
