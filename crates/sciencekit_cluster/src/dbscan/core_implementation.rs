//! The configured DBSCAN estimator (spec `dbscan`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::builder::SKDBSCANBuilder;
use super::config::{SKDBSCANAlgorithm, SKDBSCANConfiguration, SKDBSCANMetric};

/// Density-Based Spatial Clustering of Applications with Noise, constructed
/// only through [`SKDBSCANBuilder`].
///
/// The estimator is immutable configuration; fitting it returns the distinct
/// fitted `SKDBSCANModel`, which owns the learned assignments, core sample
/// indices and nearest-core assignment state.
#[derive(Debug, Clone, Copy)]
pub struct SKDBSCAN<F: SKFloat> {
    pub(super) configuration: SKDBSCANConfiguration<F>,
}

impl<F: SKFloat> SKDBSCAN<F> {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKDBSCANBuilder<F> {
        SKDBSCANBuilder::new()
    }

    /// Inclusive neighbourhood radius.
    pub fn eps(&self) -> F {
        self.configuration.eps
    }

    /// Minimum number of neighbours (the point itself included) for a core
    /// sample.
    pub fn minimum_samples(&self) -> usize {
        self.configuration.minimum_samples
    }

    /// Configured distance metric.
    pub fn metric(&self) -> SKDBSCANMetric {
        self.configuration.metric
    }

    /// Requested neighbourhood-search algorithm (may be `Auto`).
    pub fn algorithm(&self) -> SKDBSCANAlgorithm {
        self.configuration.algorithm
    }

    /// Maximum number of rows in a spatial-index leaf.
    pub fn leaf_size(&self) -> usize {
        self.configuration.leaf_size
    }

    /// Backend kind recorded in every operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }

    /// Builds the estimator from an already validated configuration.
    pub(super) fn from_configuration(configuration: SKDBSCANConfiguration<F>) -> Self {
        Self { configuration }
    }
}
