//! Configured Principal Component Analysis estimator (`SKPCA`, spec
//! `principal-component-analysis`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
};

use super::builder::SKPCABuilder;
use super::config::{SKPCAConfiguration, SKPCASolver};

/// Principal Component Analysis, constructible only through [`SKPCABuilder`].
///
/// `SKPCA` holds immutable configuration; fitting produces a distinct fitted
/// model owning the learned mean, principal components and variance state.
#[derive(Debug, Clone, Copy)]
pub struct SKPCA<F: SKFloat> {
    pub(super) configuration: SKPCAConfiguration<F>,
}

impl<F: SKFloat> SKPCA<F> {
    /// Opens a [`SKPCABuilder`] initialized with the documented defaults.
    pub fn build() -> SKPCABuilder<F> {
        SKPCABuilder::new()
    }

    /// Configured component count (`None` retains `min(n_samples, n_features)`).
    pub fn number_of_components(&self) -> Option<usize> {
        self.configuration.number_of_components
    }

    /// Requested solver (`Auto` by default).
    pub fn svd_solver(&self) -> SKPCASolver {
        self.configuration.svd_solver
    }

    /// Whether transformed projections are whitened to unit variance.
    pub fn whiten(&self) -> bool {
        self.configuration.whiten
    }

    /// Configured singular-value tolerance.
    pub fn tolerance(&self) -> F {
        self.configuration.tolerance
    }

    /// Number of power iterations for the randomized SVD solver.
    pub fn number_of_power_iterations(&self) -> usize {
        self.configuration.number_of_power_iterations
    }

    /// Whether explicit sparse-to-dense materialization is permitted for
    /// dense-only solvers.
    pub fn allow_densification(&self) -> bool {
        self.configuration.allow_densification
    }

    /// Deterministic seed for the randomized projection.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
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

    pub(super) fn from_configuration(configuration: SKPCAConfiguration<F>) -> Self {
        Self { configuration }
    }
}
