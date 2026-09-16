use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKUnsupervisedFit,
    execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::kernels::{SK_AXIS_SUM_GRAIN, sk_axis_sum, sk_axis_sum_squares};

use crate::standard_scaler::builder::SKStandardScalerBuilder;
use crate::standard_scaler::model::SKStandardScalerModel;

pub struct SKStandardScaler {
    pub(super) plan: SKBuilderState,
    pub(super) with_mean: bool,
    pub(super) with_std: bool,
    pub(super) backend: SKBackendKind,
}

impl SKStandardScaler {
    pub fn build() -> SKStandardScalerBuilder {
        SKStandardScalerBuilder::new()
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
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKStandardScaler {
    type Model = SKStandardScalerModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        let (mean, scale) = standardize(self, &dense)?;
        Ok(SKStandardScalerModel::from_statistics(
            mean,
            scale,
            self.with_mean,
            self.with_std,
            self.backend,
        ))
    }
}

/// Learn per-column mean and scale under a `Fit` observability span.
pub(super) fn standardize<F: SKFloat>(
    estimator: &SKStandardScaler,
    features: &ArrayView2<F>,
) -> Result<(Array1<F>, Array1<F>), SKError> {
    let (rows_count, columns_count) = validated_shape(features)?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows: rows_count,
        columns: columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend,
    };
    sk_run_operation(attributes, || {
        let plan = resolve_fit_plan(estimator, features)?;
        let (column_sums, column_sum_squares) = column_statistics(features, plan.parallelism);
        Ok(moments_from_sums(
            &column_sums,
            &column_sum_squares,
            rows_count,
            estimator.with_mean,
            estimator.with_std,
        ))
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

/// Build a fresh execution context per operation (never at `build()` time),
/// declare the dominant kernel's grain and the `Sequential` access pattern,
/// and resolve the plan.
fn resolve_fit_plan<F: SKFloat>(
    estimator: &SKStandardScaler,
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

/// Per-column Σx and Σx², parametrized by the resolved parallelism.
fn column_statistics<F: SKFloat>(
    features: &ArrayView2<F>,
    parallelism: usize,
) -> (Array1<F>, Array1<F>) {
    let column_sums = sk_axis_sum(features, 0, parallelism);
    let column_sum_squares = sk_axis_sum_squares(features, 0, parallelism);
    (column_sums, column_sum_squares)
}

/// Derive mean and scale from the column sums: population variance via
/// `E[x²] − mean²`, clamping tiny cancellation negatives to zero.
/// Zero-variance columns (or `with_std = false`) yield a scale of one so
/// `transform` divides safely; `with_mean = false` yields a zero mean.
fn moments_from_sums<F: SKFloat>(
    column_sums: &Array1<F>,
    column_sum_squares: &Array1<F>,
    rows_count: usize,
    with_mean: bool,
    with_std: bool,
) -> (Array1<F>, Array1<F>) {
    let rows_as_scalar = F::from(rows_count).expect("row count fits in scalar");
    let columns_count = column_sums.len();
    let mut mean = Array1::zeros(columns_count);
    let mut scale = Array1::from_elem(columns_count, F::one());
    for index in 0..columns_count {
        let column_mean = column_sums[index] / rows_as_scalar;
        let variance = population_variance(column_sum_squares[index], column_mean, rows_as_scalar);
        let (mean_value, scale_value) = finalize_column(column_mean, variance, with_mean, with_std);
        mean[index] = mean_value;
        scale[index] = scale_value;
    }
    (mean, scale)
}

/// Population variance from Σx² and the mean, flooring float noise at zero.
fn population_variance<F: SKFloat>(sum_squares: F, column_mean: F, rows_as_scalar: F) -> F {
    let variance = (sum_squares / rows_as_scalar) - column_mean * column_mean;
    if variance > F::zero() {
        variance
    } else {
        F::zero()
    }
}

/// Apply the `with_mean` / `with_std` flags to one column: a disabled mean
/// becomes zero, a disabled or non-positive spread becomes a unit scale.
pub(super) fn finalize_column<F: SKFloat>(
    column_mean: F,
    variance: F,
    with_mean: bool,
    with_std: bool,
) -> (F, F) {
    let mean_value = mask_or_zero(column_mean, with_mean);
    let scale_value = scale_or_one(variance.sqrt(), with_std);
    (mean_value, scale_value)
}

/// Keep the value when the flag is set, else the additive identity.
fn mask_or_zero<F: SKFloat>(value: F, enabled: bool) -> F {
    if enabled { value } else { F::zero() }
}

/// Keep a positive divisor when enabled, else the multiplicative identity.
fn scale_or_one<F: SKFloat>(divisor: F, enabled: bool) -> F {
    if enabled && divisor > F::zero() {
        divisor
    } else {
        F::one()
    }
}
