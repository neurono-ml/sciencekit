use ndarray::{Array1, Array2, azip};
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat,
    SKOperationAttributes, SKOperationKind, sk_run_operation,
};

/// Fitted min/max scaler: the learned per-column minimum and range plus the
/// affine multiplier and the configured target range.
///
/// Distinct type from the estimator (`SKMinMaxScaler`): the estimator is
/// immutable configuration, the model is the learned state and the export
/// boundary. The learned minimum/range are the single source of truth — the
/// affine `scale` is refreshed from them by every constructor and update — so
/// an incremental update converges to the same parameters as a batch fit.
#[derive(Debug, Clone)]
pub struct SKMinMaxScalerModel<F> {
    pub(super) sample_count: u64,
    pub(super) minimum: Array1<F>,
    pub(super) maximum: Array1<F>,
    pub(super) data_range: Array1<F>,
    pub(super) scale: Array1<F>,
    pub(super) feature_minimum: F,
    pub(super) feature_maximum: F,
    pub(super) clip: bool,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKMinMaxScalerModel<F> {
    /// Build a fitted model from learned extrema. Crate-internal: models are
    /// produced by `fit`/`fit_streaming`, never constructed by consumers.
    pub(crate) fn from_extrema(
        sample_count: u64,
        minimum: Array1<F>,
        maximum: Array1<F>,
        feature_range: (F, F),
        clip: bool,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        debug_assert!(sample_count > 0, "a fitted model covers at least one row");
        let (feature_minimum, feature_maximum) = feature_range;
        let data_range = derive_range(&minimum, &maximum);
        let scale = derive_scale(&data_range, feature_minimum, feature_maximum);
        Self {
            sample_count,
            minimum,
            maximum,
            data_range,
            scale,
            feature_minimum,
            feature_maximum,
            clip,
            backend,
            execution_intent,
        }
    }

    /// The number of rows the model has learned from.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Per-column learned minimum.
    pub fn minimum(&self) -> &Array1<F> {
        &self.minimum
    }

    /// Per-column learned range (`maximum - minimum`).
    pub fn range(&self) -> &Array1<F> {
        &self.data_range
    }

    /// Per-column affine multiplier (ones for zero-range columns).
    pub fn scale(&self) -> &Array1<F> {
        &self.scale
    }

    /// The configured target range.
    pub fn feature_range(&self) -> (F, F) {
        (self.feature_minimum, self.feature_maximum)
    }

    /// Whether transformed values are clamped to the target range.
    pub fn clip(&self) -> bool {
        self.clip
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }
}

/// `maximum - minimum` per column.
pub(crate) fn derive_range<F: SKFloat>(minimum: &Array1<F>, maximum: &Array1<F>) -> Array1<F> {
    minimum
        .iter()
        .zip(maximum.iter())
        .map(|(&lower, &upper)| upper - lower)
        .collect()
}

/// The affine multiplier `(r_max - r_min) / range`, with a unit multiplier for a
/// zero-range (constant) column so its transformed values stay finite.
pub(crate) fn derive_scale<F: SKFloat>(
    data_range: &Array1<F>,
    feature_minimum: F,
    feature_maximum: F,
) -> Array1<F> {
    let span = feature_maximum - feature_minimum;
    data_range
        .iter()
        .map(|&range| {
            if range > F::zero() {
                span / range
            } else {
                F::one()
            }
        })
        .collect()
}

impl<F: SKFloat> SKFeatureTransformer<F> for SKMinMaxScalerModel<F> {
    type Output = Array2<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let dense = view.as_dense()?;
        if dense.ncols() != self.minimum.len() {
            return Err(SKError::shape_mismatch_2d(
                dense.nrows(),
                self.minimum.len(),
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
                azip!((value in &mut row, minimum in &self.minimum, scale in &self.scale) {
                    let mut scaled = (*value - *minimum) * *scale + self.feature_minimum;
                    if self.clip {
                        scaled = scaled.max(self.feature_minimum).min(self.feature_maximum);
                    }
                    *value = scaled;
                });
            }
            Ok(output)
        })
    }
}
