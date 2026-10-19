use ndarray::{Array1, Array2, azip};
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat,
    SKOperationAttributes, SKOperationKind, sk_run_operation,
};

use crate::robust_scaler::core_implementation::SKRobustScalerConfiguration;

/// Fitted robust scaler: the learned per-column center and scale (already
/// masked by the configured flags) plus the configuration.
///
/// Distinct type from the estimator (`SKRobustScaler`): the estimator is
/// immutable configuration, the model is the learned state and the export
/// boundary. The center and scale are the single source of truth for the
/// transform; both are refreshed by every constructor and update.
#[derive(Debug, Clone)]
pub struct SKRobustScalerModel<F> {
    pub(super) sample_count: u64,
    pub(super) center: Array1<F>,
    pub(super) scale: Array1<F>,
    pub(super) configuration: SKRobustScalerConfiguration<F>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKRobustScalerModel<F> {
    /// Build a fitted model from learned center and scale. Crate-internal:
    /// models are produced by `fit`/`fit_streaming`, never constructed by
    /// consumers.
    pub(crate) fn from_quantiles(
        sample_count: u64,
        center: Array1<F>,
        scale: Array1<F>,
        configuration: SKRobustScalerConfiguration<F>,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        debug_assert!(sample_count > 0, "a fitted model covers at least one row");
        Self {
            sample_count,
            center,
            scale,
            configuration,
            backend,
            execution_intent,
        }
    }

    /// The number of rows the model has learned from.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Per-column learned center (zeros when fitted with `with_centering = false`).
    pub fn center(&self) -> &Array1<F> {
        &self.center
    }

    /// Per-column learned scale (ones for zero-spread columns or
    /// `with_scaling = false`).
    pub fn scale(&self) -> &Array1<F> {
        &self.scale
    }

    /// Whether the transform centers on the median.
    pub fn with_centering(&self) -> bool {
        self.configuration.with_centering
    }

    /// Whether the transform scales by the interquantile range.
    pub fn with_scaling(&self) -> bool {
        self.configuration.with_scaling
    }

    /// The `(low, high)` percentile pair the scale is built from.
    pub fn quantile_range(&self) -> (F, F) {
        self.configuration.quantile_range()
    }

    /// Whether the scale is divided by the standard-normal interquantile span.
    pub fn unit_variance(&self) -> bool {
        self.configuration.unit_variance
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }
}

impl<F: SKFloat> SKFeatureTransformer<F> for SKRobustScalerModel<F> {
    type Output = Array2<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let dense = view.as_dense()?;
        if dense.ncols() != self.center.len() {
            return Err(SKError::shape_mismatch_2d(
                dense.nrows(),
                self.center.len(),
                dense.nrows(),
                dense.ncols(),
            ));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count: dense.nrows(),
            columns_count: dense.ncols(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let mut output = dense.to_owned();
            for mut row in output.rows_mut() {
                azip!((value in &mut row, center in &self.center, scale in &self.scale) {
                    *value = (*value - *center) / *scale;
                });
            }
            Ok(output)
        })
    }
}
