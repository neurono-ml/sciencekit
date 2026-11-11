//! Validated configuration for `SKIncrementalPCA` (spec `incremental-pca`).

use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode};

/// Default batch size used when `fit` chunks an in-memory matrix.
pub(super) const DEFAULT_BATCH_SIZE: usize = 1024;

/// Default pass count used when `fit_streaming` drives a reopenable provider.
pub(super) const DEFAULT_MAXIMUM_NUMBER_OF_EPOCHS: usize = 1;

/// Validated configuration carried by a built incremental PCA estimator.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(super) struct SKIncrementalPCAConfiguration {
    /// Optional component count (`None` retains up to the feature count).
    pub(super) number_of_components: Option<usize>,
    /// Row block size used when `fit` chunks a full design matrix.
    pub(super) batch_size: usize,
    /// Whether transformed projections are scaled to unit variance.
    pub(super) whiten: bool,
    /// Epoch budget for reopenable streaming providers.
    pub(super) maximum_number_of_epochs: usize,
    /// Math backend recorded in spans and dispatched per operation.
    pub(super) backend: SKBackendKind,
    /// Shared builder state holding the execution intent.
    pub(super) builder_state: SKBuilderState,
}

#[allow(dead_code)]
impl SKIncrementalPCAConfiguration {
    /// Constructs configuration with the documented defaults.
    pub(super) fn default_configuration() -> Self {
        Self {
            number_of_components: None,
            batch_size: DEFAULT_BATCH_SIZE,
            whiten: false,
            maximum_number_of_epochs: DEFAULT_MAXIMUM_NUMBER_OF_EPOCHS,
            backend: SKBackendKind::Faer,
            builder_state: SKBuilderState::new(),
        }
    }

    /// Returns the configured execution mode.
    pub(super) fn execution_mode(&self) -> SKExecutionMode {
        self.builder_state.execution_intent()
    }
}
