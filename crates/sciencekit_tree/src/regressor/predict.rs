//! Regressor prediction.

use ndarray::Array1;
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    sk_run_operation,
};

use crate::design::sk_sparse_row_value;

use super::model::SKDecisionTreeRegressorModel;

impl<F: SKFloat> SKDecisionTreeRegressorModel<F> {
    /// Predict one response per row, reading the leaf mean or median.
    pub fn predict_values<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_dimensions(&data)?;
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
            sk_validate_values(&data)?;
            match data {
                SKDataView::Dense(dense) => Ok(Array1::from_vec(
                    (0..rows_count)
                        .into_par_iter()
                        .map(|row| {
                            let leaf = self.tree.leaf_index_for(|feature| dense[[row, feature]]);
                            self.tree.node(leaf).expect("leaf exists").value()[0]
                        })
                        .collect(),
                )),
                SKDataView::Sparse(sparse) => Ok(Array1::from_vec(
                    (0..rows_count)
                        .into_par_iter()
                        .map(|row| {
                            let sparse_row = sparse.outer_view(row).expect("row within bounds");
                            let leaf = self.tree.leaf_index_for(|feature| {
                                sk_sparse_row_value(&sparse_row, feature)
                            });
                            self.tree.node(leaf).expect("leaf exists").value()[0]
                        })
                        .collect(),
                )),
                _ => Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                }),
            }
        })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKDecisionTreeRegressorModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_values(features)
    }
}

fn sk_dimensions<F>(features: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    match features {
        SKDataView::Dense(dense) => Ok((dense.nrows(), dense.ncols())),
        SKDataView::Sparse(sparse) => Ok((sparse.rows(), sparse.cols())),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

fn sk_validate_values<F: SKFloat>(data: &SKDataView<'_, F>) -> Result<(), SKError> {
    let columns = match data {
        SKDataView::Dense(dense) => dense.ncols(),
        SKDataView::Sparse(sparse) => sparse.cols(),
        _ => 0,
    };
    match data {
        SKDataView::Dense(dense) => {
            if let Some(position) = dense
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
                .map(|(position, _)| position)
            {
                let row = position / columns;
                let column = position % columns;
                return Err(SKError::NonFiniteValue {
                    position: row * columns + column,
                });
            }
        }
        SKDataView::Sparse(sparse) => {
            for (row, sparse_row) in sparse.outer_iterator().enumerate() {
                let bad = sparse_row
                    .indices()
                    .iter()
                    .zip(sparse_row.data().iter())
                    .find(|(_, value)| !value.is_finite());
                if let Some((&column, _)) = bad {
                    return Err(SKError::NonFiniteValue {
                        position: row * columns + column,
                    });
                }
            }
        }
        _ => {}
    }
    Ok(())
}
