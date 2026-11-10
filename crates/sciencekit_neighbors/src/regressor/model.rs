//! The fitted k-nearest-neighbour regressor model
//! (spec `nearest-neighbor-estimators`).

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::neighbors::{
    SKFittedSearch, SKNeighborsAlgorithm, SKNeighborsConfiguration, SKNeighborsMetric,
    SKNeighborsWeights, SKSearchStrategy,
};

/// Fitted regressor state: the stored training design, the resolved search
/// strategy and each training row's response.
#[derive(Debug, Clone)]
pub struct SKKNeighborsRegressorModel<F: SKFloat> {
    pub(super) search: SKFittedSearch<F>,
    pub(super) configuration: SKNeighborsConfiguration,
    pub(super) responses: Array1<F>,
}

impl<F: SKFloat> SKKNeighborsRegressorModel<F> {
    /// The strategy resolved at fit time (never `Auto`).
    pub fn search_strategy(&self) -> SKSearchStrategy {
        self.search.strategy
    }

    /// The number of training rows the model holds.
    pub fn training_rows(&self) -> usize {
        self.search.training_rows()
    }

    /// The number of features the model was fitted on.
    pub fn number_of_features(&self) -> usize {
        self.search.number_of_features()
    }

    /// The representation the training design is stored in (`dense` or `csr`).
    pub fn training_representation(&self) -> &'static str {
        self.search.training.representation()
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

    /// Backend kind recorded by every model operation.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Execution intent carried from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }
}
