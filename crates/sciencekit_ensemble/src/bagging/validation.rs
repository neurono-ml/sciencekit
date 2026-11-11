//! Input-view validation and target canonicalization for bagging estimators.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKLabelTable, SKTargetView, sk_canonicalize_labels,
};
use sprs::CsMatView;

/// Validate that a fit feature view has at least one row and column, and uses
/// dense or CSR storage.
pub(super) fn sk_bagging_fit_dimensions<F>(
    features: &SKDataView<'_, F>,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = sk_extract_view_shape(features)?;
    if rows == 0 {
        return Err(SKError::EmptyInput { operation: "fit" });
    }
    if columns == 0 {
        return Err(SKError::shape_mismatch_2d(rows, 1, rows, 0));
    }
    Ok((rows, columns))
}

/// Validate that a prediction feature view has at least one row, matches
/// `expected_columns`, and uses dense or CSR storage.
#[allow(dead_code)]
pub(super) fn sk_bagging_predict_dimensions<F>(
    features: &SKDataView<'_, F>,
    expected_columns: usize,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = sk_extract_view_shape(features)?;
    if rows == 0 {
        return Err(SKError::EmptyInput {
            operation: "predict",
        });
    }
    if columns != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows,
            expected_columns,
            rows,
            columns,
        ));
    }
    Ok((rows, columns))
}

fn sk_extract_view_shape<F>(features: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    match features {
        SKDataView::Dense(dense) => Ok((dense.nrows(), dense.ncols())),
        SKDataView::Sparse(sparse) => {
            if !sparse.is_csr() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR with to_csr",
                });
            }
            Ok((sparse.rows(), sparse.cols()))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

/// Reject any non-finite value in a dense or sparse CSR view at its flattened
/// row-major position.
pub(super) fn sk_validate_bagging_finite_features<F: SKFloat>(
    features: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            let columns = dense.ncols();
            dense
                .rows()
                .into_iter()
                .enumerate()
                .try_for_each(|(row, values)| {
                    if let Some(column) = values.iter().position(|value| !value.is_finite()) {
                        return Err(SKError::NonFiniteValue {
                            position: row * columns + column,
                        });
                    }
                    Ok(())
                })
        }
        SKDataView::Sparse(sparse) => sk_validate_finite_sparse_csr(sparse),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

fn sk_validate_finite_sparse_csr<F: SKFloat>(sparse: &CsMatView<'_, F>) -> Result<(), SKError> {
    let columns = sparse.cols();
    sparse
        .outer_iterator()
        .enumerate()
        .try_for_each(|(row, sparse_row)| {
            let non_finite = sparse_row
                .indices()
                .iter()
                .zip(sparse_row.data().iter())
                .find(|(_, value)| !value.is_finite());
            if let Some((&column, _)) = non_finite {
                return Err(SKError::NonFiniteValue {
                    position: row * columns + column,
                });
            }
            Ok(())
        })
}

/// Canonicalize integer or nominal classification targets, rejecting continuous
/// targets.
pub(super) fn sk_canonicalize_bagging_targets<F: SKFloat>(
    targets: &SKTargetView<'_, F>,
) -> Result<(Vec<usize>, SKLabelTable), SKError> {
    match targets {
        SKTargetView::Nominal(labels) => Ok(sk_canonicalize_labels(labels)),
        SKTargetView::Integer(values) => {
            let owned: Vec<String> = values.iter().map(ToString::to_string).collect();
            let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
            Ok(sk_canonicalize_labels(&borrowed))
        }
        SKTargetView::Continuous(_) => Err(SKError::UnsupportedRepresentation {
            representation: "continuous",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
    }
}

/// Collect distinct canonical class indices in first-occurrence order.
pub(super) fn sk_first_occurrence_classes(sampled_classes: &[usize]) -> Vec<usize> {
    let mut distinct = Vec::new();
    sampled_classes.iter().for_each(|&class_index| {
        if !distinct.contains(&class_index) {
            distinct.push(class_index);
        }
    });
    distinct
}
