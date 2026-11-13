//! The validated search configuration shared by both neighbor estimators
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode};

use super::{SKNeighborsAlgorithm, SKNeighborsMetric, SKNeighborsWeights};

/// The search surface both estimators carry, validated at build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKNeighborsConfiguration {
    /// Number of neighbours consulted per query row.
    pub(crate) nearest_neighbors_count: usize,
    /// How neighbours are weighted.
    pub(crate) weights: SKNeighborsWeights,
    /// The requested search algorithm, possibly `Auto`.
    pub(crate) algorithm: SKNeighborsAlgorithm,
    /// The distance metric.
    pub(crate) metric: SKNeighborsMetric,
    /// Maximum number of rows in a tree leaf.
    pub(crate) leaf_size: usize,
    /// Backend kind recorded in every operation span.
    pub(crate) backend: SKBackendKind,
    /// Execution intent carried from the builder.
    pub(crate) execution_mode: SKExecutionMode,
}

impl SKNeighborsConfiguration {
    /// Validate the values that do not depend on the training data.
    pub(crate) fn validate(&self) -> Result<(), SKError> {
        if self.nearest_neighbors_count == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "nearest_neighbors_count",
                reason: "must be strictly positive".to_string(),
            });
        }
        if self.leaf_size == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "leaf_size",
                reason: "must be strictly positive".to_string(),
            });
        }
        Ok(())
    }
}
