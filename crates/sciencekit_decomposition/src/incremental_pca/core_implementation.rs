//! Configured Incremental Principal Component Analysis estimator
//! (`SKIncrementalPCA`, spec `incremental-pca`).

use std::marker::PhantomData;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
};

use super::builder::SKIncrementalPCABuilder;
use super::config::SKIncrementalPCAConfiguration;

/// Incremental Principal Component Analysis, constructible only through
/// [`SKIncrementalPCABuilder`].
///
/// `SKIncrementalPCA` holds immutable configuration; fitting produces a
/// distinct fitted model owning the running Chan-Golub-LeVeque covariance
/// accumulator and finalized principal components.
#[derive(Debug, Clone, Copy)]
pub struct SKIncrementalPCA<F: SKFloat> {
    pub(super) configuration: SKIncrementalPCAConfiguration,
    scalar: PhantomData<fn() -> F>,
}

impl<F: SKFloat> SKIncrementalPCA<F> {
    /// Opens a [`SKIncrementalPCABuilder`] initialized with the documented
    /// defaults.
    pub fn build() -> SKIncrementalPCABuilder<F> {
        SKIncrementalPCABuilder::new()
    }

    /// Configured component count (`None` retains up to the feature count).
    pub fn number_of_components(&self) -> Option<usize> {
        self.configuration.number_of_components
    }

    /// Configured row batch size used when `fit` chunks an in-memory matrix.
    pub fn batch_size(&self) -> usize {
        self.configuration.batch_size
    }

    /// Whether transformed projections are scaled to unit variance.
    pub fn whiten(&self) -> bool {
        self.configuration.whiten
    }

    /// Maximum number of streaming passes over a reopenable provider.
    pub fn maximum_number_of_epochs(&self) -> usize {
        self.configuration.maximum_number_of_epochs
    }

    /// Backend kind recorded in operation spans and transferred to fitted
    /// models.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode()
    }

    /// Resolves the estimator's execution intent against `context`.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.configuration.builder_state.resolve_plan(context)
    }

    pub(super) fn from_configuration(configuration: SKIncrementalPCAConfiguration) -> Self {
        Self {
            configuration,
            scalar: PhantomData,
        }
    }
}
