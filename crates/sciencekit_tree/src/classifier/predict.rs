//! Classifier prediction.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};
use sprs::CsMatView;

use crate::design::sk_sparse_row_value;
use crate::node::SKTree;

use super::model::SKDecisionTreeClassifierModel;

impl<F: SKFloat> SKDecisionTreeClassifierModel<F> {
    /// Class probabilities for every row, one column per fitted class.
    pub fn probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_prediction_dimensions(&data)?;
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
            sk_validate_prediction_values(&data)?;
            let mut probabilities = Array2::<F>::zeros((rows_count, self.number_of_classes));
            let leaves = sk_leaf_indices(&self.tree, &data)?;
            for (row, leaf) in leaves.iter().enumerate() {
                let distribution = self.tree.node(*leaf).expect("leaf exists").value();
                for (class, probability) in distribution.iter().enumerate() {
                    probabilities[[row, class]] = *probability;
                }
            }
            Ok(probabilities)
        })
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKDecisionTreeClassifierModel<F> {
    type Error = SKError;

    /// Predict canonical class indices.
    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let probabilities = self.probabilities(features)?;
        Ok(Array1::from_shape_fn(probabilities.nrows(), |row| {
            let mut best = 0usize;
            for class in 1..probabilities.ncols() {
                if probabilities[[row, class]] > probabilities[[row, best]] {
                    best = class;
                }
            }
            best as i64
        }))
    }

    /// Predict class probabilities, one column per fitted class.
    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.probabilities(features)
    }
}

/// The leaf every row reaches, in input order, under either representation.
fn sk_leaf_indices<F: SKFloat>(
    tree: &SKTree<F>,
    data: &SKDataView<'_, F>,
) -> Result<Vec<usize>, SKError> {
    match data {
        SKDataView::Dense(dense) => Ok((0..dense.nrows())
            .into_par_iter()
            .map(|row| tree.leaf_index_for(|feature| dense[[row, feature]]))
            .collect()),
        SKDataView::Sparse(sparse) => Ok((0..sparse.rows())
            .into_par_iter()
            .map(|row| {
                let sparse_row = sparse.outer_view(row).expect("row within bounds");
                tree.leaf_index_for(|feature| sk_sparse_row_value(&sparse_row, feature))
            })
            .collect()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

/// Validate the prediction shape.
fn sk_prediction_dimensions<F>(features: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    match features {
        SKDataView::Dense(dense) => Ok((dense.nrows(), dense.ncols())),
        SKDataView::Sparse(sparse) => Ok((sparse.rows(), sparse.cols())),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

/// Reject non-finite stored values before any traversal.
fn sk_validate_prediction_values<F: SKFloat>(data: &SKDataView<'_, F>) -> Result<(), SKError> {
    let columns = match data {
        SKDataView::Dense(dense) => dense.ncols(),
        SKDataView::Sparse(sparse) => sparse.cols(),
        _ => 0,
    };
    match data {
        SKDataView::Dense(dense) => {
            for (row, values) in dense.rows().into_iter().enumerate() {
                for (column, &value) in values.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(SKError::NonFiniteValue {
                            position: row * columns + column,
                        });
                    }
                }
            }
        }
        SKDataView::Sparse(sparse) => sk_validate_sparse(sparse, columns)?,
        _ => {}
    }
    Ok(())
}

/// Reject non-finite sparse entries at their flattened position.
fn sk_validate_sparse<F: SKFloat>(
    sparse: &CsMatView<'_, F>,
    columns: usize,
) -> Result<(), SKError> {
    for (row, sparse_row) in sparse.outer_iterator().enumerate() {
        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data().iter()) {
            if !value.is_finite() {
                return Err(SKError::NonFiniteValue {
                    position: row * columns + column,
                });
            }
        }
    }
    Ok(())
}
