use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKUnsupervisedFit,
    execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::kernels::{SK_AXIS_SUM_GRAIN, sk_axis_min_max};

use crate::minmax_scaler::builder::SKMinMaxScalerBuilder;
use crate::minmax_scaler::model::SKMinMaxScalerModel;

/// Immutable, builder-configured min/max scaler estimator. The learned state
/// lives in the distinct [`SKMinMaxScalerModel`]; the estimator carries only the
/// configuration and the carried backend kind.
pub struct SKMinMaxScaler<F> {
    pub(super) plan: SKBuilderState,
    pub(super) feature_minimum: F,
    pub(super) feature_maximum: F,
    pub(super) clip: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKMinMaxScaler<F> {
    pub fn build() -> SKMinMaxScalerBuilder<F> {
        SKMinMaxScalerBuilder::new()
    }

    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }

    /// The backend kind this estimator records in its operation spans and
    /// transfers into its fitted models.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured target range transformed features are mapped onto.
    pub fn feature_range(&self) -> (F, F) {
        (self.feature_minimum, self.feature_maximum)
    }

    /// Whether transformed values are clamped to the target range.
    pub fn clip(&self) -> bool {
        self.clip
    }
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKMinMaxScaler<F> {
    type Model = SKMinMaxScalerModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        let (minimum, maximum) = column_extrema(self, &dense)?;
        Ok(SKMinMaxScalerModel::from_extrema(
            dense.nrows() as u64,
            minimum,
            maximum,
            self.feature_range(),
            self.clip,
            self.backend,
            self.execution_intent(),
        ))
    }
}

/// Learn per-column extrema under a `Fit` observability span: the running
/// minimum and maximum every model constructor needs.
pub(super) fn column_extrema<F: SKFloat>(
    estimator: &SKMinMaxScaler<F>,
    features: &ArrayView2<F>,
) -> Result<(Array1<F>, Array1<F>), SKError> {
    let (rows_count, columns_count) = validated_shape(features)?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend,
    };
    sk_run_operation(attributes, || {
        let plan = resolve_fit_plan(estimator, features)?;
        Ok(sk_axis_min_max(features, 0, plan.parallelism))
    })
}

/// Reject empty inputs before any element is processed.
fn validated_shape<F>(features: &ArrayView2<F>) -> Result<(usize, usize), SKError> {
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
    estimator: &SKMinMaxScaler<F>,
    features: &ArrayView2<F>,
) -> Result<SKExecutionPlan, SKError> {
    let mut context = SKExecutionContext::real();
    context.dataset_size_bytes = features.len() as u64 * size_of::<F>() as u64;
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = Some(features.len() as u64);
    context.access_pattern = SKAccessPattern::Sequential;
    context.grain = SK_AXIS_SUM_GRAIN;
    estimator.resolve_plan(&context)
}
