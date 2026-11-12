//! The configured nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat,
};

use crate::knn::builder::SKKNNImputerBuilder;
use crate::knn::weights::SKImputationWeights;

/// Multivariate missing-value imputer constructed only through its builder
/// (spec `nearest-neighbor-imputation`).
///
/// The estimator is immutable configuration; fitting returns the distinct
/// [`crate::knn::SKKNNImputerModel`], which owns the training design, its
/// missingness mask and the per-column donor indices.
pub struct SKKNNImputer<F> {
    pub(in crate::knn) plan: SKBuilderState,
    pub(in crate::knn) missing_values: F,
    pub(in crate::knn) nearest_neighbors_count: usize,
    pub(in crate::knn) weights: SKImputationWeights,
    pub(in crate::knn) add_indicator: bool,
    pub(in crate::knn) keep_empty_features: bool,
    pub(in crate::knn) backend: SKBackendKind,
}

impl<F: SKFloat> SKKNNImputer<F> {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKKNNImputerBuilder<F> {
        SKKNNImputerBuilder::new()
    }

    /// The configured missing-value marker.
    pub fn missing_values(&self) -> F {
        self.missing_values
    }

    /// The number of donors consulted per missing cell.
    pub fn nearest_neighbors_count(&self) -> usize {
        self.nearest_neighbors_count
    }

    /// The configured donor weighting scheme.
    pub fn weights(&self) -> SKImputationWeights {
        self.weights
    }

    /// Whether all-missing columns are kept as `0.0` instead of dropped.
    pub fn keep_empty_features(&self) -> bool {
        self.keep_empty_features
    }

    /// Whether the transform stacks a missingness indicator per fit-time
    /// missing feature.
    pub fn add_indicator(&self) -> bool {
        self.add_indicator
    }

    /// The backend kind this estimator records in its operation spans and
    /// transfers into its fitted model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Resolve the accumulated intent against a context into a concrete plan.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }
}
