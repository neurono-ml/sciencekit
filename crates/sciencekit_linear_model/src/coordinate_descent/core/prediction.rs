//! Sparse-edition prediction for one alpha index (spec
//! `coordinate-descent-solver`, Decision 10).

use ndarray::{Array1, Axis};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, sk_run_operation,
};
use sciencekit_math::sk_csr_dense_product;

use super::definition::SKCoordinateDescentModelCore;

impl<F: SKFloat> SKCoordinateDescentModelCore<F> {
    /// Predict responses for one explicit `alpha` index.
    ///
    /// Accepts dense and sparse features; a sparse design is multiplied by the
    /// selected coefficient column without densifying. A column count differing
    /// from the fitted design is rejected with `ShapeMismatch`.
    pub(crate) fn predict_with_alpha_index<'a, X>(
        &self,
        index: usize,
        features: X,
    ) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = match data {
            SKDataView::Dense(dense) => dense.dim(),
            SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
        if columns_count != self.feature_count() {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                self.feature_count(),
                rows_count,
                columns_count,
            ));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let intercept = self.intercepts[index];
            let coefficient_row = self.coefficients.row(index);
            match data {
                SKDataView::Dense(dense) => {
                    Ok(dense.dot(&coefficient_row).mapv(|value| value + intercept))
                }
                SKDataView::Sparse(sparse) => {
                    let coefficient_column = coefficient_row.insert_axis(Axis(1));
                    let product = sk_csr_dense_product(&sparse, &coefficient_column);
                    Ok(product.column(0).mapv(|value| value + intercept))
                }
                _ => Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                }),
            }
        })
    }
}
