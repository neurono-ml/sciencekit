//! The fitted DBSCAN model (spec `dbscan`, decision 6).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{SKBackendKind, SKClusterLabel, SKExecutionMode, SKFloat};

use super::config::{SKDBSCANConfiguration, SKDBSCANMetric, SKDBSCANSearchStrategy};
use super::expansion::SKDBSCANExpansion;

/// Fitted DBSCAN state: typed training assignments, core-sample indices,
/// cluster count, fitted `eps` and the producing run's configuration.
#[derive(Debug, Clone)]
pub struct SKDBSCANModel<F: SKFloat> {
    pub(super) labels: Array1<SKClusterLabel>,
    pub(super) core_sample_indices: Array1<usize>,
    pub(super) number_of_clusters: usize,
    pub(super) number_of_features: usize,
    pub(super) eps: F,
    pub(super) minimum_samples: usize,
    pub(super) metric: SKDBSCANMetric,
    pub(super) search_strategy: SKDBSCANSearchStrategy,
    pub(super) training_representation: &'static str,
    pub(super) backend: SKBackendKind,
    pub(super) execution_mode: SKExecutionMode,
}

impl<F: SKFloat> SKDBSCANModel<F> {
    /// Builds the model from a completed DBSCAN expansion.
    pub(super) fn from_expansion(
        expansion: SKDBSCANExpansion,
        number_of_features: usize,
        training_representation: &'static str,
        configuration: SKDBSCANConfiguration<F>,
    ) -> Self {
        Self {
            labels: expansion.labels,
            core_sample_indices: expansion.core_sample_indices,
            number_of_clusters: expansion.number_of_clusters,
            number_of_features,
            eps: configuration.eps,
            minimum_samples: configuration.minimum_samples,
            metric: configuration.metric,
            search_strategy: expansion.strategy,
            training_representation,
            backend: configuration.backend,
            execution_mode: configuration.execution_mode,
        }
    }

    /// Typed training assignment (`Cluster` or `Noise`), one entry per training
    /// row.
    pub fn labels(&self) -> ArrayView1<'_, SKClusterLabel> {
        self.labels.view()
    }

    /// Sorted ascending training row indices of all core samples.
    pub fn core_sample_indices(&self) -> ArrayView1<'_, usize> {
        self.core_sample_indices.view()
    }

    /// Number of distinct clusters discovered (excluding noise).
    pub fn number_of_clusters(&self) -> usize {
        self.number_of_clusters
    }

    /// Number of features the model was fitted on.
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// Inclusive neighbourhood radius used by the fit and by nearest-core
    /// assignment.
    pub fn eps(&self) -> F {
        self.eps
    }

    /// Minimum neighbour count used to identify core samples.
    pub fn minimum_samples(&self) -> usize {
        self.minimum_samples
    }

    /// Distance metric used by the fit and by nearest-core assignment.
    pub fn metric(&self) -> SKDBSCANMetric {
        self.metric
    }

    /// Resolved neighbourhood-search strategy (never `Auto`).
    pub fn search_strategy(&self) -> SKDBSCANSearchStrategy {
        self.search_strategy
    }

    /// Representation of the training input (`"dense"` or `"csr"`).
    pub fn training_representation(&self) -> &'static str {
        self.training_representation
    }

    /// Backend kind recorded by the producing estimator and by every model
    /// operation.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent carried from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }
}
