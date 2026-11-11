//! Feature-subset routing and canonical class alignment for classifier members.

use ndarray::{Array2, Axis, Slice};
use sciencekit_common::{SKClassifierPredictor, SKDataView, SKError, SKFloat};

use super::super::error::SKBaggingError;
use super::super::model::SKBaggingClassifierMember;
use super::super::sampling::{
    sk_gather_dense_submatrix, sk_gather_sparse_submatrix, sk_is_identity_features,
    sk_project_dense_rows, sk_project_sparse_rows,
};

/// Route contiguous rows `row_start..row_end` through `feature_subset` into a
/// classifier member and return canonical `(chunk_rows, number_of_classes)`
/// probabilities.
#[allow(dead_code)]
pub(in crate::bagging) fn sk_predict_classifier_member_chunk<F, M>(
    member: &SKBaggingClassifierMember<M>,
    features: &SKDataView<'_, F>,
    row_start: usize,
    row_end: usize,
    feature_subset: &[usize],
    number_of_classes: usize,
) -> Result<Array2<F>, SKBaggingError<M::Error>>
where
    F: SKFloat,
    M: SKClassifierPredictor<F>,
{
    let chunk_rows = row_end - row_start;
    match member {
        SKBaggingClassifierMember::ConstantClass { class_index } => {
            let mut matrix = Array2::zeros((chunk_rows, number_of_classes));
            matrix.column_mut(*class_index).mapv_inplace(|_| F::one());
            Ok(matrix)
        }
        SKBaggingClassifierMember::Fitted {
            model,
            local_to_ensemble,
        } => {
            let local_probabilities = match features {
                SKDataView::Dense(dense) => {
                    if sk_is_identity_features(feature_subset, dense.ncols()) {
                        let slice = dense.slice_axis(Axis(0), Slice::from(row_start..row_end));
                        model
                            .predict_probabilities(slice)
                            .map_err(SKBaggingError::Base)?
                    } else {
                        let projected =
                            sk_project_dense_rows(dense, row_start, row_end, feature_subset);
                        model
                            .predict_probabilities(projected.view())
                            .map_err(SKBaggingError::Base)?
                    }
                }
                SKDataView::Sparse(sparse) => {
                    if row_start == 0
                        && row_end == sparse.rows()
                        && sk_is_identity_features(feature_subset, sparse.cols())
                    {
                        model
                            .predict_probabilities(*sparse)
                            .map_err(SKBaggingError::Base)?
                    } else {
                        let projected =
                            sk_project_sparse_rows(sparse, row_start, row_end, feature_subset);
                        model
                            .predict_probabilities(projected.view())
                            .map_err(SKBaggingError::Base)?
                    }
                }
                _ => unreachable!("dimensions already validated"),
            };
            sk_align_member_probabilities(
                local_probabilities,
                chunk_rows,
                local_to_ensemble,
                number_of_classes,
            )
            .map_err(SKBaggingError::Central)
        }
    }
}

/// Route arbitrary `rows` (such as OOB rows) through `feature_subset` into a
/// classifier member.
#[allow(dead_code)]
pub(in crate::bagging) fn sk_predict_classifier_member_rows<F, M>(
    member: &SKBaggingClassifierMember<M>,
    features: &SKDataView<'_, F>,
    rows: &[usize],
    feature_subset: &[usize],
    number_of_classes: usize,
) -> Result<Array2<F>, SKBaggingError<M::Error>>
where
    F: SKFloat,
    M: SKClassifierPredictor<F>,
{
    match member {
        SKBaggingClassifierMember::ConstantClass { class_index } => {
            let mut matrix = Array2::zeros((rows.len(), number_of_classes));
            matrix.column_mut(*class_index).mapv_inplace(|_| F::one());
            Ok(matrix)
        }
        SKBaggingClassifierMember::Fitted {
            model,
            local_to_ensemble,
        } => {
            let local_probabilities = match features {
                SKDataView::Dense(dense) => {
                    let submatrix = sk_gather_dense_submatrix(dense, rows, feature_subset);
                    model
                        .predict_probabilities(submatrix.view())
                        .map_err(SKBaggingError::Base)?
                }
                SKDataView::Sparse(sparse) => {
                    let submatrix = sk_gather_sparse_submatrix(sparse, rows, feature_subset);
                    model
                        .predict_probabilities(submatrix.view())
                        .map_err(SKBaggingError::Base)?
                }
                _ => unreachable!("dimensions already validated"),
            };
            sk_align_member_probabilities(
                local_probabilities,
                rows.len(),
                local_to_ensemble,
                number_of_classes,
            )
            .map_err(SKBaggingError::Central)
        }
    }
}

fn sk_align_member_probabilities<F: SKFloat>(
    local_probabilities: Array2<F>,
    expected_rows: usize,
    local_to_ensemble: &[usize],
    number_of_classes: usize,
) -> Result<Array2<F>, SKError> {
    let (found_rows, found_columns) = local_probabilities.dim();
    if found_rows != expected_rows {
        return Err(SKError::shape_mismatch_2d(
            expected_rows,
            number_of_classes,
            found_rows,
            found_columns,
        ));
    }
    if found_columns == local_to_ensemble.len() {
        let mut aligned = Array2::zeros((expected_rows, number_of_classes));
        local_to_ensemble
            .iter()
            .enumerate()
            .for_each(|(local_column, &ensemble_column)| {
                aligned
                    .column_mut(ensemble_column)
                    .assign(&local_probabilities.column(local_column));
            });
        return Ok(aligned);
    }
    if found_columns == number_of_classes {
        return Ok(local_probabilities);
    }
    Err(SKError::shape_mismatch_2d(
        expected_rows,
        local_to_ensemble.len(),
        found_rows,
        found_columns,
    ))
}
