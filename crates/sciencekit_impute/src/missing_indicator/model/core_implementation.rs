//! The fitted indicator state: the mask scope, the marker and the carried
//! configuration (spec `missing-indicator`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::missing_indicator::scope::SKMissingFeatureScope;

/// The learned state of a [`crate::missing_indicator::SKMissingIndicator`]:
/// the feature columns the mask covers, the marker and the carried backend
/// kind. The sole bearer of the fitted state, `Send + Sync` by construction.
pub struct SKMissingIndicatorModel<F> {
    pub(super) scope: Vec<usize>,
    pub(super) number_of_input_columns: usize,
    pub(super) features: SKMissingFeatureScope,
    pub(super) error_on_new: bool,
    pub(super) sparse_output: bool,
    pub(super) missing_values: F,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKMissingIndicatorModel<F> {
    /// The feature columns the mask covers, in ascending order.
    pub fn scope(&self) -> &[usize] {
        &self.scope
    }

    /// The number of input columns the model was fitted on.
    pub fn number_of_input_columns(&self) -> usize {
        self.number_of_input_columns
    }

    /// The number of mask columns the transform emits.
    pub fn number_of_output_columns(&self) -> usize {
        self.scope.len()
    }

    /// The feature scope requested at build time.
    pub fn features(&self) -> SKMissingFeatureScope {
        self.features
    }

    /// Whether missingness fit never saw is rejected at transform.
    pub fn error_on_new(&self) -> bool {
        self.error_on_new
    }

    /// Whether a dense input yields a CSR mask.
    pub fn sparse_output(&self) -> bool {
        self.sparse_output
    }

    /// The missing-value marker used by the mask.
    pub fn missing_values(&self) -> F {
        self.missing_values
    }

    /// The backend kind recorded in this model's operation spans.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }
}
