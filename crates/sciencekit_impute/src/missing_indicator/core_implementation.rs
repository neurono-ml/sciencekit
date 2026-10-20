//! The `SKMissingIndicator` estimator shell (spec `missing-indicator`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKUnsupervisedFit,
};

use crate::missing_indicator::builder::SKMissingIndicatorBuilder;
use crate::missing_indicator::model::SKMissingIndicatorModel;
use crate::missing_indicator::scope::SKMissingFeatureScope;

/// Learn which feature columns carry missing cells and emit a `0.0`/`1.0` mask
/// for them (spec `missing-indicator`). The learned scope lives in the distinct
/// model returned by `fit`.
pub struct SKMissingIndicator<F> {
    pub(in crate::missing_indicator) plan: SKBuilderState,
    pub(in crate::missing_indicator) features: SKMissingFeatureScope,
    pub(in crate::missing_indicator) error_on_new: bool,
    pub(in crate::missing_indicator) sparse_output: bool,
    pub(in crate::missing_indicator) missing_values: F,
    pub(in crate::missing_indicator) backend: SKBackendKind,
}

impl<F: SKFloat> SKMissingIndicator<F> {
    /// The builder of this estimator; the only construction path.
    pub fn build() -> SKMissingIndicatorBuilder<F> {
        SKMissingIndicatorBuilder::new()
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Resolve the accumulated intent against a context into a concrete plan.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }

    /// The backend kind this estimator records in its operation spans and
    /// transfers into its fitted models.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Which feature columns the mask covers.
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

    /// The configured missing-value marker.
    pub fn missing_values(&self) -> F {
        self.missing_values
    }
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKMissingIndicator<F> {
    type Model = SKMissingIndicatorModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, _features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "the missing indicator is not implemented yet",
        })
    }
}
