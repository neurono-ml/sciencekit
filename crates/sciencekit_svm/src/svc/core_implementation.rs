//! Configured `SKSVC` estimator and hyperparameter accessors
//! (spec `support-vector-machines`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};
use sciencekit_math::{SKKernel, SKKernelGamma};

use super::builder::SKSVCBuilder;
use super::weights::SKClassWeight;

/// Internal validated configuration carried by [`SKSVC`].
#[derive(Debug, Clone)]
pub(crate) struct SKSVCConfiguration<F: SKFloat> {
    pub(crate) regularization_c: F,
    pub(crate) kernel: SKKernel,
    pub(crate) degree: u32,
    pub(crate) gamma: SKKernelGamma<F>,
    pub(crate) coef0: F,
    pub(crate) tolerance: F,
    pub(crate) maximum_number_of_iterations: usize,
    pub(crate) cache_size: usize,
    pub(crate) shrinking: bool,
    pub(crate) class_weight: SKClassWeight<F>,
    pub(crate) probability: bool,
    pub(crate) random_seed: u64,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_mode: SKExecutionMode,
}

/// Configured C-Support-Vector Classifier (`SKSVC`).
///
/// Construct exclusively through [`SKSVC::build`].
#[derive(Debug, Clone)]
pub struct SKSVC<F: SKFloat> {
    pub(crate) configuration: SKSVCConfiguration<F>,
}

impl<F: SKFloat> SKSVC<F> {
    pub(crate) fn from_configuration(configuration: SKSVCConfiguration<F>) -> Self {
        Self { configuration }
    }

    /// Start building an [`SKSVC`] with documented defaults.
    pub fn build() -> SKSVCBuilder<F> {
        SKSVCBuilder::new()
    }

    /// Configured regularization parameter `C`.
    #[allow(non_snake_case)]
    pub fn C(&self) -> F {
        self.configuration.regularization_c
    }

    /// Configured kernel kind.
    pub fn kernel(&self) -> SKKernel {
        self.configuration.kernel
    }

    /// Configured polynomial degree.
    pub fn degree(&self) -> u32 {
        self.configuration.degree
    }

    /// Configured `gamma` policy.
    pub fn gamma(&self) -> SKKernelGamma<F> {
        self.configuration.gamma
    }

    /// Configured kernel offset `coef0`.
    pub fn coef0(&self) -> F {
        self.configuration.coef0
    }

    /// Configured KKT stopping tolerance.
    pub fn tolerance(&self) -> F {
        self.configuration.tolerance
    }

    /// Configured iteration budget.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.configuration.maximum_number_of_iterations
    }

    /// Configured LRU kernel-row cache budget in MiB.
    pub fn cache_size(&self) -> usize {
        self.configuration.cache_size
    }

    /// Whether active-set shrinking is enabled.
    pub fn shrinking(&self) -> bool {
        self.configuration.shrinking
    }

    /// Configured class-weighting policy.
    pub fn class_weight(&self) -> &SKClassWeight<F> {
        &self.configuration.class_weight
    }

    /// Whether Platt + pairwise-coupling probability calibration is enabled.
    pub fn probability(&self) -> bool {
        self.configuration.probability
    }

    /// Configured deterministic seed for cross-validated Platt calibration.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// Configured math backend kind.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution mode.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }
}
