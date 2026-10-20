//! The `SKSimpleImputer` estimator shell (spec `simple-imputer`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionMode, SKExecutionPlan, SKFloat,
    SKOperationAttributes, SKOperationKind, SKUnsupervisedFit, sk_run_operation,
};

use crate::simple_imputer::builder::SKSimpleImputerBuilder;
use crate::simple_imputer::model::SKSimpleImputerModel;
use crate::simple_imputer::statistics::column_statistics;
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
        self.strategy
    }

    /// The configured missing-value marker.
    pub fn missing_values(&self) -> F {
        self.missing_values
    }

    /// The configured constant fill, if any.
    pub fn fill_value(&self) -> Option<F> {
        self.fill_value
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
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKSimpleImputer<F> {
    type Model = SKSimpleImputerModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let dense = view.as_dense()?;
        let (rows_count, columns_count) = validated_shape(&dense)?;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        };
        let statistics = sk_run_operation(attributes, || {
            let plan = resolve_fit_plan(self, &dense)?;
            debug_assert!(plan.parallelism >= 1);
            Ok::<_, SKError>(column_statistics(
                &dense,
                self.strategy,
                self.missing_values,
                self.fill_value,
            ))
        })?;
        let dropped_columns = empty_columns(&statistics, self.keep_empty_features);
        Ok(SKSimpleImputerModel::from_state(
            statistics,
            dropped_columns,
            columns_count,
            self,
        ))
    }
}

/// The indices of the columns that produced no statistic, kept only when the
/// estimator retains empty features.
fn empty_columns<F: SKFloat>(
    statistics: &ndarray::Array1<F>,
    keep_empty_features: bool,
) -> Vec<usize> {
    if keep_empty_features {
        return Vec::new();
    }
    statistics
        .iter()
        .enumerate()
        .filter(|(_, statistic)| statistic.is_nan())
        .map(|(index, _)| index)
        .collect()
}

/// Reject empty inputs before any element is processed.
fn validated_shape<F>(features: &ndarray::ArrayView2<F>) -> Result<(usize, usize), SKError> {
    let (rows_count, columns_count) = features.dim();
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    Ok((rows_count, columns_count))
}

/// Build a fresh execution context per operation, declare the reduction grain
/// and the `Sequential` access pattern, and resolve the plan.
fn resolve_fit_plan<F: SKFloat>(
    estimator: &SKSimpleImputer<F>,
    features: &ndarray::ArrayView2<F>,
) -> Result<SKExecutionPlan, SKError> {
    let mut context = sciencekit_common::SKExecutionContext::real();
    context.dataset_size_bytes = features.len() as u64 * size_of::<F>() as u64;
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = Some(features.len() as u64);
    context.access_pattern = sciencekit_common::execution::SKAccessPattern::Sequential;
    context.grain = sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;
    estimator.resolve_plan(&context)
}
