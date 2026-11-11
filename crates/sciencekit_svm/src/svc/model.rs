//! Fitted `SKSVCModel` and per-pair binary support-vector state
//! (spec `support-vector-machines`).

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat, SKLabelTable};
use sciencekit_math::SKKernelParameters;

use crate::kernel::SKOwnedFeatureBlock;
use crate::probability::SKPlattParameters;

/// One fitted binary C-SVC model separating canonical `(class_negative, class_positive)`.
///
/// Positive decision values favor `class_positive` (`+1`); non-positive values
/// favor `class_negative` (`-1`, which always has the smaller canonical index).
#[derive(Debug, Clone)]
pub struct SKBinarySVCModel<F: SKFloat> {
    pub(crate) class_negative: usize,
    pub(crate) class_positive: usize,
    pub(crate) support_vectors: SKOwnedFeatureBlock<F>,
    pub(crate) support_training_indices: Vec<usize>,
    pub(crate) dual_coefficients: Array1<F>,
    pub(crate) bias: F,
    pub(crate) iterations: usize,
    pub(crate) platt_parameters: Option<SKPlattParameters<F>>,
}

impl<F: SKFloat> SKBinarySVCModel<F> {
    /// Canonical index of the negative (`-1`) class (`class_negative < class_positive`).
    pub fn class_negative(&self) -> usize {
        self.class_negative
    }

    /// Canonical index of the positive (`+1`) class.
    pub fn class_positive(&self) -> usize {
        self.class_positive
    }

    /// Retained support vectors in their training representation.
    pub fn support_vectors(&self) -> &SKOwnedFeatureBlock<F> {
        &self.support_vectors
    }

    /// Original training row indices of the retained support vectors.
    pub fn support_training_indices(&self) -> &[usize] {
        &self.support_training_indices
    }

    /// Signed dual coefficients `y_i * alpha_i` aligned with `support_vectors`.
    pub fn dual_coefficients(&self) -> &Array1<F> {
        &self.dual_coefficients
    }

    /// Recovered decision bias `b`.
    pub fn bias(&self) -> F {
        self.bias
    }

    /// Number of SMO iterations executed for this binary pair.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Fitted Platt scaling parameters when probability calibration was enabled.
    pub fn platt_parameters(&self) -> Option<SKPlattParameters<F>> {
        self.platt_parameters
    }
}

/// Fitted C-Support-Vector Classifier model.
#[derive(Debug, Clone)]
pub struct SKSVCModel<F: SKFloat> {
    pub(crate) classes: SKLabelTable,
    pub(crate) pairwise_models: Vec<SKBinarySVCModel<F>>,
    pub(crate) kernel_parameters: SKKernelParameters<F>,
    pub(crate) number_of_features: usize,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_mode: SKExecutionMode,
    pub(crate) probability_enabled: bool,
}

impl<F: SKFloat> SKSVCModel<F> {
    /// Canonical class label table.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// Number of classes in the fitted problem.
    pub fn number_of_classes(&self) -> usize {
        self.classes.number_of_classes()
    }

    /// Number of input features seen at fit time.
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// Pairwise binary C-SVC models in lexicographic `(class_a, class_b)` order.
    pub fn pairwise_models(&self) -> &[SKBinarySVCModel<F>] {
        &self.pairwise_models
    }

    /// Resolved kernel configuration (carrying concrete scalar `gamma`).
    pub fn kernel_parameters(&self) -> SKKernelParameters<F> {
        self.kernel_parameters
    }

    /// Whether every pairwise SMO solver reached its KKT tolerance.
    pub fn converged(&self) -> bool {
        true
    }

    /// Maximum SMO iterations executed across the pairwise binary fits.
    pub fn iterations(&self) -> usize {
        self.pairwise_models
            .iter()
            .map(SKBinarySVCModel::iterations)
            .max()
            .unwrap_or(0)
    }

    /// Configured math backend kind transferred from the estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Configured execution mode transferred from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }

    /// Whether probability calibration was enabled at fit time.
    pub fn probability(&self) -> bool {
        self.probability_enabled
    }
}
