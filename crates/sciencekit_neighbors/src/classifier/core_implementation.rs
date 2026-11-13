//! The configured k-nearest-neighbour classifier
//! (spec `nearest-neighbor-estimators`).

use std::marker::PhantomData;

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::SKKNeighborsClassifierBuilder;
use crate::neighbors::{
    SKNeighborsAlgorithm, SKNeighborsConfiguration, SKNeighborsMetric, SKNeighborsWeights,
};

/// k-nearest-neighbour classifier, constructed only through its builder.
///
/// The estimator is immutable configuration; fitting returns the distinct
/// [`super::SKKNeighborsClassifierModel`], which owns the training design, the
/// resolved search strategy and the class table.
#[derive(Debug, Clone, Copy)]
pub struct SKKNeighborsClassifier<F: SKFloat> {
    pub(super) configuration: SKNeighborsConfiguration,
    pub(super) marker: PhantomData<F>,
}

impl<F: SKFloat> SKKNeighborsClassifier<F> {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKKNeighborsClassifierBuilder<F> {
        SKKNeighborsClassifierBuilder::new()
    }

    /// Number of neighbours consulted per query row.
    pub fn nearest_neighbors_count(&self) -> usize {
        self.configuration.nearest_neighbors_count
    }

    /// The neighbour weighting scheme.
    pub fn weights(&self) -> SKNeighborsWeights {
        self.configuration.weights
    }

    /// The requested search algorithm (may be `Auto`).
    pub fn algorithm(&self) -> SKNeighborsAlgorithm {
        self.configuration.algorithm
    }

    /// The distance metric.
    pub fn metric(&self) -> SKNeighborsMetric {
        self.configuration.metric
    }

    /// The maximum number of rows in a tree leaf.
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
    pub(super) fn from_configuration(configuration: SKNeighborsConfiguration) -> Self {
        Self {
            configuration,
            marker: PhantomData,
        }
    }
}
