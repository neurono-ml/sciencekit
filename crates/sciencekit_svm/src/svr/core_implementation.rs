//! Configured `SKSVR` estimator and hyperparameter accessors
//! (spec `support-vector-machines`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};
use sciencekit_math::{SKKernel, SKKernelGamma};

use super::builder::SKSVRBuilder;

/// Internal validated configuration carried by [`SKSVR`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKSVRConfiguration<F: SKFloat> {
    pub(crate) regularization_c: F,
    pub(crate) epsilon: F,
    pub(crate) kernel: SKKernel,
    pub(crate) degree: u32,
    pub(crate) gamma: SKKernelGamma<F>,
    pub(crate) coef0: F,
    pub(crate) tolerance: F,
    pub(crate) maximum_number_of_iterations: usize,
    pub(crate) cache_size: usize,
    pub(crate) shrinking: bool,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_mode: SKExecutionMode,
}

/// Configured Epsilon-Support-Vector Regressor (`SKSVR`).
///
/// Construct exclusively through [`SKSVR::build`].
#[derive(Debug, Clone, Copy)]
pub struct SKSVR<F: SKFloat> {
    pub(crate) configuration: SKSVRConfiguration<F>,
}

impl<F: SKFloat> SKSVR<F> {
    pub(crate) fn from_configuration(configuration: SKSVRConfiguration<F>) -> Self {
        Self { configuration }
    }

    /// Start building an [`SKSVR`] with documented defaults.
    pub fn build() -> SKSVRBuilder<F> {
        SKSVRBuilder::new()
    }

    /// Configured regularization parameter `C`.
    #[allow(non_snake_case)]
    pub fn C(&self) -> F {
        self.configuration.regularization_c
    }

    /// Configured epsilon-insensitive half-width `epsilon`.
    pub fn epsilon(&self) -> F {
        self.configuration.epsilon
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

    /// Configured math backend kind.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution mode.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }
}
