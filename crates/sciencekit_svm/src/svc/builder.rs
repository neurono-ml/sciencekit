//! Builder-only construction for [`SKSVC`] (spec `support-vector-machines`).

use sciencekit_common::builders::sk_validate_hyperparameter;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
};
use sciencekit_math::{SKKernel, SKKernelGamma, sk_validate_kernel_hyperparameters};

use super::core_implementation::{SKSVC, SKSVCConfiguration};
use super::weights::{SKClassWeight, sk_validate_static_class_weight};

/// Builder for [`SKSVC`].
pub struct SKSVCBuilder<F: SKFloat> {
    state: SKBuilderState,
    configuration: SKSVCConfiguration<F>,
}

impl<F: SKFloat> SKSVCBuilder<F> {
    /// Create a builder carrying the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            configuration: SKSVCConfiguration {
                regularization_c: F::one(),
                kernel: SKKernel::RadialBasisFunction,
                degree: 3,
                gamma: SKKernelGamma::Scale,
                coef0: F::zero(),
                tolerance: F::from(1e-3).expect("1e-3 fits SKFloat"),
                maximum_number_of_iterations: 10_000,
                cache_size: 200,
                shrinking: true,
                class_weight: SKClassWeight::None,
                probability: false,
                random_seed: 0,
                backend: SKBackendKind::Faer,
                execution_mode: SKExecutionMode::Automatic,
            },
        }
    }

    /// Set the regularization parameter `C` (default `1.0`).
    #[allow(non_snake_case)]
    pub fn C(mut self, value: F) -> Self {
        self.configuration.regularization_c = value;
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

    /// Set the class-weighting policy (default `None`).
    pub fn class_weight(mut self, value: SKClassWeight<F>) -> Self {
        self.configuration.class_weight = value;
        self
    }

    /// Enable or disable Platt + pairwise-coupling probability calibration (default `false`).
    pub fn probability(mut self, value: bool) -> Self {
        self.configuration.probability = value;
        self
    }

    /// Set the deterministic seed for cross-validated Platt calibration (default `0`).
    pub fn random_seed(mut self, value: u64) -> Self {
        self.configuration.random_seed = value;
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
        self.state.execution_mode(mode);
        self.configuration.execution_mode = self.state.execution_intent();
        self
    }
}

impl<F: SKFloat> Default for SKSVCBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKSVC<F>> for SKSVCBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self.configuration.execution_mode = self.state.execution_intent();
        self
    }

    fn build(self) -> Result<SKSVC<F>, SKError> {
        let config = &self.configuration;
        sk_validate_hyperparameter(
            "C",
            config.regularization_c > F::zero() && config.regularization_c.is_finite(),
            "regularization strength C must be positive and finite",
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
        sk_validate_static_class_weight(&config.class_weight)?;
        Ok(SKSVC::from_configuration(self.configuration))
    }
}
