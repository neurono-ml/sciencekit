//! Shared feature-matrix and target-vector validation helpers for fit and
//! prediction.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKLabelTable, SKTargetView, sk_canonicalize_labels,
};
use sprs::{CsMatView, CsVecView};

/// Validate that a fit feature view has at least one row and column, and uses a
/// supported representation.
pub(super) fn sk_fit_dimensions<F>(
    features: &SKDataView<'_, F>,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = match features {
        SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
        SKDataView::Sparse(sparse) => {
            if !sparse.is_csr() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR with to_csr",
                });
            }
            (sparse.rows(), sparse.cols())
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            });
        }
    };
    if rows == 0 {
        return Err(SKError::EmptyInput { operation: "fit" });
    }
    if columns == 0 {
        return Err(SKError::shape_mismatch_2d(rows, 1, rows, 0));
    }
    Ok((rows, columns))
}

/// Validate that a prediction feature view has at least one row, matches the
/// fitted feature count, and uses a supported representation.
pub(super) fn sk_predict_dimensions<F>(
    features: &SKDataView<'_, F>,
    expected_columns: usize,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = match features {
        SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
        SKDataView::Sparse(sparse) => {
            if !sparse.is_csr() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR with to_csr",
                });
            }
            (sparse.rows(), sparse.cols())
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            });
        }
    };
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

/// Reject any non-finite value in a dense or sparse CSR view at its flattened
/// row-major position.
pub(super) fn sk_validate_finite_features<F: SKFloat>(
    features: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            let columns = dense.ncols();
            for (row, values) in dense.rows().into_iter().enumerate() {
                if let Some(column) = values.iter().position(|value| !value.is_finite()) {
                    return Err(SKError::NonFiniteValue {
                        position: row * columns + column,
                    });
                }
            }
            Ok(())
        }
        SKDataView::Sparse(sparse) => sk_validate_finite_sparse(sparse),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

fn sk_validate_finite_sparse<F: SKFloat>(sparse: &CsMatView<'_, F>) -> Result<(), SKError> {
    let columns = sparse.cols();
    for (row, sparse_row) in sparse.outer_iterator().enumerate() {
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
    }
    Ok(())
}

/// Canonicalize integer or nominal classification targets, rejecting continuous
/// targets.
pub(super) fn sk_canonicalize_targets<F: SKFloat>(
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

/// Read one feature value from a CSR row view, treating an absent coordinate as
/// an observed zero.
pub(super) fn sk_sparse_feature_value<F: SKFloat>(row: &CsVecView<'_, F>, feature: usize) -> F {
    match row.indices().binary_search(&feature) {
        Ok(position) => row.data()[position],
        Err(_) => F::zero(),
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
