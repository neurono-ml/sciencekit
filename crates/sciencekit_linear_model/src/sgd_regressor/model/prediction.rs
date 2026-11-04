//! Prediction for the fitted SGD regressor model (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, Axis};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    sk_run_operation,
};
use sciencekit_math::sk_csr_dense_product;

use super::core_implementation::SKSGDRegressorModel;

impl<F: SKFloat> SKSGDRegressorModel<F> {
    /// Predict one continuous response per row, `X·coefficients + intercept`.
    fn predict_responses<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = dimensions(&data);
        if columns_count != self.feature_count() {
            return Err(self.feature_mismatch(rows_count, columns_count));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let responses = match data {
                SKDataView::Dense(dense) => dense
                    .dot(&self.coefficients)
                    .mapv(|value| value + self.intercept),
                SKDataView::Sparse(sparse) => {
                    let weights = self.coefficients.view().insert_axis(Axis(1));
                    let product = sk_csr_dense_product(&sparse, &weights);
                    product.column(0).mapv(|value| value + self.intercept)
                }
                _ => return Err(unknown_representation()),
            };
            Ok(responses)
        })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKSGDRegressorModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_responses(features)
    }
}

/// The row and column counts of a feature view.
pub(crate) fn dimensions<F>(data: &SKDataView<'_, F>) -> (usize, usize) {
    match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// The error for a representation prediction cannot consume.
fn unknown_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}
