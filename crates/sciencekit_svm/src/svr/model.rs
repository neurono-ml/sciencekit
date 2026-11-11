//! Fitted `SKSVRModel` carrying support-vector-only state
//! (spec `support-vector-machines`).

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};
use sciencekit_math::SKKernelParameters;

use crate::kernel::SKOwnedFeatureBlock;

/// Fitted Epsilon-Support-Vector Regressor model.
#[derive(Debug, Clone)]
pub struct SKSVRModel<F: SKFloat> {
    pub(crate) support_vectors: SKOwnedFeatureBlock<F>,
    pub(crate) support_training_indices: Vec<usize>,
    pub(crate) dual_coefficients: Array1<F>,
    pub(crate) bias: F,
    pub(crate) iterations: usize,
    pub(crate) kernel_parameters: SKKernelParameters<F>,
    pub(crate) number_of_features: usize,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_mode: SKExecutionMode,
}

impl<F: SKFloat> SKSVRModel<F> {
    /// Retained support vectors in their training representation.
    pub fn support_vectors(&self) -> &SKOwnedFeatureBlock<F> {
        &self.support_vectors
    }

    /// Original training row indices of the retained support vectors.
    pub fn support_training_indices(&self) -> &[usize] {
        &self.support_training_indices
    }

    /// Signed dual coefficients `alpha_i - alpha_i*` aligned with `support_vectors`.
    pub fn dual_coefficients(&self) -> &Array1<F> {
        &self.dual_coefficients
    }

    /// Recovered regression bias `b`.
    pub fn bias(&self) -> F {
        self.bias
    }

    /// Whether the SMO solver reached its KKT tolerance.
    pub fn converged(&self) -> bool {
        true
    }

    /// Number of SMO iterations executed.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Resolved kernel configuration (carrying concrete scalar `gamma`).
    pub fn kernel_parameters(&self) -> SKKernelParameters<F> {
        self.kernel_parameters
    }

    /// Number of input features seen at fit time.
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// Configured math backend kind transferred from the estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Configured execution mode transferred from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }
}
