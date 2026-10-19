use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat,
};

use crate::standard_scaler::core_implementation::finalize_column;

/// Fitted standard scaler: the running Welford moments plus the masked mean
/// and derived scale the last update produced.
///
/// Distinct type from the estimator (`SKStandardScaler`): the estimator is
/// immutable configuration, the model is the learned state and the export
/// boundary. Moments are the single source of truth — `mean`/`scale` are
/// refreshed from them by every constructor and every `partial_fit` — so
/// incremental and batch fits converge to the same statistics.
#[derive(Debug, Clone)]
pub struct SKStandardScalerModel<F> {
    pub(super) sample_count: u64,
    pub(super) moments_mean: Array1<F>,
    pub(super) sum_squared_deviations: Array1<F>,
    pub(super) mean: Array1<F>,
    pub(super) scale: Array1<F>,
    pub(super) with_mean: bool,
    pub(super) with_std: bool,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKStandardScalerModel<F> {
    /// Build a fitted model from learned moments: per-column mean, sum of
    /// squared deviations and row count. Crate-internal: models are produced
    /// by `fit`/`fit_streaming`, never constructed directly by consumers.
    pub(crate) fn from_moments(
        sample_count: u64,
        moments_mean: Array1<F>,
        sum_squared_deviations: Array1<F>,
        with_mean: bool,
        with_std: bool,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        debug_assert!(sample_count > 0, "a fitted model covers at least one row");
        let (mean, scale) = derive_statistics(
            &moments_mean,
            &sum_squared_deviations,
            sample_count,
            with_mean,
            with_std,
        );
        Self {
            sample_count,
            moments_mean,
            sum_squared_deviations,
            mean,
            scale,
            with_mean,
            with_std,
            backend,
            execution_intent,
        }
    }

    /// The number of rows the model has learned from, across `fit` and every
    /// `partial_fit`.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Per-column means (zeros when fitted with `with_mean = false`).
    pub fn mean(&self) -> &Array1<F> {
        &self.mean
    }

    /// Per-column scales (ones for zero-variance columns or `with_std = false`).
    pub fn scale(&self) -> &Array1<F> {
        &self.scale
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }
}

/// Derive the public masked mean and scale from running moments. Shared by
/// the constructor and the incremental `partial_fit`.
pub(crate) fn derive_statistics<F: SKFloat>(
    moments_mean: &Array1<F>,
    sum_squared_deviations: &Array1<F>,
    sample_count: u64,
    with_mean: bool,
    with_std: bool,
) -> (Array1<F>, Array1<F>) {
    let count_as_scalar = F::from(sample_count).expect("row count fits in scalar");
    let mean: Array1<F> = moments_mean
        .iter()
        .zip(sum_squared_deviations.iter())
        .map(|(&column_mean, &deviations)| {
            let variance = deviations / count_as_scalar;
            finalize_column(column_mean, variance, with_mean, with_std).0
        })
        .collect();
    let scale: Array1<F> = moments_mean
        .iter()
        .zip(sum_squared_deviations.iter())
        .map(|(&column_mean, &deviations)| {
            let variance = deviations / count_as_scalar;
            finalize_column(column_mean, variance, with_mean, with_std).1
        })
        .collect();
    (mean, scale)
}

impl<F: SKFloat> SKFeatureTransformer<F> for SKStandardScalerModel<F> {
    type Output = Array2<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        use sciencekit_common::{SKOperationAttributes, SKOperationKind, sk_run_operation};

        let view = features.try_into()?;
        let dense = view.as_dense()?;
        if dense.ncols() != self.mean.len() {
            return Err(SKError::shape_mismatch_2d(
                dense.nrows(),
                self.mean.len(),
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
                ndarray::azip!((value in &mut row, mean in &self.mean, scale in &self.scale) {
                    let centered = if self.with_mean {
                        *value - *mean
                    } else {
                        *value
                    };
                    *value = if self.with_std {
                        centered / *scale
                    } else {
                        centered
                    };
                });
            }
            Ok(output)
        })
    }
}
