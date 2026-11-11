//! Feature-subset routing for regressor base models.

use ndarray::{Array1, Axis, Slice};
use sciencekit_common::{SKDataView, SKError, SKFloat, SKRegressorPredictor};

use super::super::error::SKBaggingError;
use super::super::sampling::{
    sk_gather_dense_submatrix, sk_gather_sparse_submatrix, sk_is_identity_features,
    sk_project_dense_rows, sk_project_sparse_rows,
};

/// Route contiguous rows `row_start..row_end` through `feature_subset` into a
/// regressor model.
pub(in crate::bagging) fn sk_predict_regressor_chunk<F, M>(
    estimator: &M,
    features: &SKDataView<'_, F>,
    row_start: usize,
    row_end: usize,
    feature_subset: &[usize],
) -> Result<Array1<F>, SKBaggingError<M::Error>>
where
    F: SKFloat,
    M: SKRegressorPredictor<F>,
{
    let chunk_rows = row_end - row_start;
    let predictions = match features {
        SKDataView::Dense(dense) => {
            if sk_is_identity_features(feature_subset, dense.ncols()) {
                let slice = dense.slice_axis(Axis(0), Slice::from(row_start..row_end));
                estimator.predict(slice).map_err(SKBaggingError::Base)?
            } else {
                let projected = sk_project_dense_rows(dense, row_start, row_end, feature_subset);
                estimator
                    .predict(projected.view())
                    .map_err(SKBaggingError::Base)?
            }
        }
        SKDataView::Sparse(sparse) => {
            if row_start == 0
                && row_end == sparse.rows()
                && sk_is_identity_features(feature_subset, sparse.cols())
            {
                estimator.predict(*sparse).map_err(SKBaggingError::Base)?
            } else {
                let projected = sk_project_sparse_rows(sparse, row_start, row_end, feature_subset);
                estimator
                    .predict(projected.view())
                    .map_err(SKBaggingError::Base)?
            }
        }
        _ => unreachable!("dimensions already validated"),
    };
    if predictions.len() != chunk_rows {
        return Err(SKBaggingError::Central(SKError::shape_mismatch_2d(
            chunk_rows,
            1,
            predictions.len(),
            1,
        )));
    }
    if let Some(position) = predictions.iter().position(|value| !value.is_finite()) {
        return Err(SKBaggingError::Central(SKError::NonFiniteValue {
            position,
        }));
    }
    Ok(predictions)
}

/// Route arbitrary `rows` (such as OOB rows) through `feature_subset` into a
/// regressor model.
pub(in crate::bagging) fn sk_predict_regressor_rows<F, M>(
    estimator: &M,
    features: &SKDataView<'_, F>,
    rows: &[usize],
    feature_subset: &[usize],
) -> Result<Array1<F>, SKBaggingError<M::Error>>
where
    F: SKFloat,
    M: SKRegressorPredictor<F>,
{
    let predictions = match features {
        SKDataView::Dense(dense) => {
            let submatrix = sk_gather_dense_submatrix(dense, rows, feature_subset);
            estimator
                .predict(submatrix.view())
                .map_err(SKBaggingError::Base)?
        }
        SKDataView::Sparse(sparse) => {
            let submatrix = sk_gather_sparse_submatrix(sparse, rows, feature_subset);
            estimator
                .predict(submatrix.view())
                .map_err(SKBaggingError::Base)?
        }
        _ => unreachable!("dimensions already validated"),
    };
    if predictions.len() != rows.len() {
        return Err(SKBaggingError::Central(SKError::shape_mismatch_2d(
            rows.len(),
            1,
            predictions.len(),
            1,
        )));
    }
    if let Some(position) = predictions.iter().position(|value| !value.is_finite()) {
        return Err(SKBaggingError::Central(SKError::NonFiniteValue {
            position,
        }));
    }
    Ok(predictions)
}
