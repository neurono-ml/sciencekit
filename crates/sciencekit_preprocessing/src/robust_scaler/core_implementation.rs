use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKUnsupervisedFit,
    execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use crate::robust_scaler::builder::SKRobustScalerBuilder;
use crate::robust_scaler::model::SKRobustScalerModel;
use crate::robust_scaler::quantiles::{
    column_quantile_statistics, standard_normal_interquantile_span,
};

/// The immutable per-scaler options shared by the estimator and its model:
/// the two masking flags, the quantile range and the unit-variance flag.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKRobustScalerConfiguration<F> {
    pub(super) with_centering: bool,
    pub(super) with_scaling: bool,
    pub(super) quantile_low: F,
    pub(super) quantile_high: F,
    pub(super) unit_variance: bool,
}

impl<F: SKFloat> SKRobustScalerConfiguration<F> {
    /// The `(low, high)` percentile pair the scale is built from.
    pub(super) fn quantile_range(&self) -> (F, F) {
        (self.quantile_low, self.quantile_high)
    }

    /// The standard-normal interquantile span used as the unit-variance
    /// divisor.
    pub(super) fn unit_variance_span(&self) -> F {
        F::from(standard_normal_interquantile_span(
            self.quantile_low.to_f64().unwrap_or(25.0),
            self.quantile_high.to_f64().unwrap_or(75.0),
        ))
        .unwrap_or(F::one())
    }
}

/// Immutable, builder-configured robust scaler estimator. The learned state
/// lives in the distinct [`SKRobustScalerModel`].
pub struct SKRobustScaler<F> {
    pub(super) plan: SKBuilderState,
    pub(super) with_centering: bool,
    pub(super) with_scaling: bool,
    pub(super) quantile_low: F,
    pub(super) quantile_high: F,
    pub(super) unit_variance: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKRobustScaler<F> {
    pub fn build() -> SKRobustScalerBuilder<F> {
        SKRobustScalerBuilder::new()
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

    /// Whether the transform centers on the median.
    pub fn with_centering(&self) -> bool {
        self.with_centering
    }

    /// Whether the transform scales by the interquantile range.
    pub fn with_scaling(&self) -> bool {
        self.with_scaling
    }

    /// The `(low, high)` percentile pair the scale is built from.
    pub fn quantile_range(&self) -> (F, F) {
        (self.quantile_low, self.quantile_high)
    }

    /// Whether the scale is divided by the standard-normal interquantile span.
    pub fn unit_variance(&self) -> bool {
        self.unit_variance
    }

    pub(super) fn configuration(&self) -> SKRobustScalerConfiguration<F> {
        SKRobustScalerConfiguration {
            with_centering: self.with_centering,
            with_scaling: self.with_scaling,
            quantile_low: self.quantile_low,
            quantile_high: self.quantile_high,
            unit_variance: self.unit_variance,
        }
    }
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKRobustScaler<F> {
    type Model = SKRobustScalerModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        let (center, scale) = robust_statistics(self, &dense)?;
        Ok(SKRobustScalerModel::from_quantiles(
            dense.nrows() as u64,
            center,
            scale,
            self.configuration(),
            self.backend,
            self.execution_intent(),
        ))
    }
}

/// Learn the per-column exact center and scale under a `Fit` observability span.
pub(super) fn robust_statistics<F: SKFloat>(
    estimator: &SKRobustScaler<F>,
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
        let _plan = resolve_fit_plan(estimator, features)?;
        let (raw_center, raw_scale) =
            column_quantile_statistics(features, estimator.quantile_low, estimator.quantile_high);
        Ok(finalize_statistics(estimator, &raw_center, &raw_scale))
    })
}

/// Apply the masking flags and the unit-variance divisor to the raw quantile
/// statistics, keeping every scale positive so the output stays finite.
fn finalize_statistics<F: SKFloat>(
    estimator: &SKRobustScaler<F>,
    raw_center: &Array1<F>,
    raw_scale: &Array1<F>,
) -> (Array1<F>, Array1<F>) {
    let span = if estimator.unit_variance {
        estimator.configuration().unit_variance_span()
    } else {
        F::one()
    };
    let center: Array1<F> = raw_center
        .iter()
        .map(|&value| {
            if estimator.with_centering {
                value
            } else {
                F::zero()
            }
        })
        .collect();
    let scale: Array1<F> = raw_scale
        .iter()
        .map(|&value| {
            let scaled = if estimator.unit_variance {
                value / span
            } else {
                value
            };
            if estimator.with_scaling && scaled > F::zero() {
                scaled
            } else {
                F::one()
            }
        })
        .collect();
    (center, scale)
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
    estimator: &SKRobustScaler<F>,
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
