use ndarray::{Array1, Array2};
use sciencekit_common::{SKBackendKind, SKDataView, SKError, SKFeatureTransformer, SKFloat};

/// Fitted standard scaler: per-column mean and scale learned by `fit`.
///
/// Distinct type from the estimator (`SKStandardScaler`): the estimator is
/// immutable configuration, the model is the learned state and the export
/// boundary. `scale` never holds zero — columns with zero variance (or the
/// `with_std = false` case) store `1`, so `transform` divides safely.
#[derive(Debug, Clone)]
pub struct SKStandardScalerModel<F> {
    mean: Array1<F>,
    scale: Array1<F>,
    with_mean: bool,
    with_std: bool,
    backend: SKBackendKind,
}

impl<F: SKFloat> SKStandardScalerModel<F> {
    /// Build a fitted model from learned per-column statistics.
    pub fn from_statistics(
        mean: Array1<F>,
        scale: Array1<F>,
        with_mean: bool,
        with_std: bool,
        backend: SKBackendKind,
    ) -> Self {
        Self {
            mean,
            scale,
            with_mean,
            with_std,
            backend,
        }
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
            rows: dense.nrows(),
            columns: dense.ncols(),
            execution_mode: sciencekit_common::SKExecutionMode::Automatic,
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
