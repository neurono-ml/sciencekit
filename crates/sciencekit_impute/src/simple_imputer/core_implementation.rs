//! The `SKSimpleImputer` estimator shell (spec `simple-imputer`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, SKExecutionPlan, SKFloat,
};

use crate::simple_imputer::builder::SKSimpleImputerBuilder;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// Fill the missing cells of a numeric feature matrix with a per-column
/// strategy statistic (spec `simple-imputer`). The learned state lives in the
/// distinct model returned by `fit`.
pub struct SKSimpleImputer<F> {
    pub(super) plan: SKBuilderState,
    pub(super) strategy: SKImputationStrategy,
    pub(super) missing_values: F,
    pub(super) fill_value: Option<F>,
    pub(super) keep_empty_features: bool,
    pub(super) add_indicator: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKSimpleImputer<F> {
    /// The builder of this estimator; the only construction path.
    pub fn build() -> SKSimpleImputerBuilder<F> {
        SKSimpleImputerBuilder::new()
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Resolve the accumulated intent against a context into a concrete plan.
    pub fn resolve_plan(
        &self,
        context: &sciencekit_common::SKExecutionContext,
    ) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }

    /// The backend kind this estimator records in its operation spans and
    /// transfers into its fitted models.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured per-column strategy statistic.
    pub fn strategy(&self) -> SKImputationStrategy {
        SKImputationStrategy::Median
    }

    /// The configured missing-value marker.
    pub fn missing_values(&self) -> F {
        F::zero()
    }

    /// The configured constant fill, if any.
    pub fn fill_value(&self) -> Option<F> {
        None
    }

    /// Whether all-missing columns are kept as `0.0` instead of dropped.
    pub fn keep_empty_features(&self) -> bool {
        true
    }

    /// Whether the transform stacks a missingness indicator per fit-time
    /// missing feature.
    pub fn add_indicator(&self) -> bool {
        true
    }
}
