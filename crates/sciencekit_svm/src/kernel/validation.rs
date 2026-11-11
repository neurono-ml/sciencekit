//! Input validation shared by `SKSVC` and `SKSVR`
//! (spec `support-vector-machines`).

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::{SKKernel, sk_precomputed_kernel_matrix};
use sprs::CompressedStorage;

/// Dimensions `(rows, columns)` of a dense or CSR feature view.
pub fn sk_svm_view_dimensions<F>(view: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    match view {
        SKDataView::Dense(dense) => Ok(dense.dim()),
        SKDataView::Sparse(sparse) => {
            if sparse.storage() != CompressedStorage::CSR {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR",
                });
            }
            Ok((sparse.rows(), sparse.cols()))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: view.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Reject NaN or infinity in a dense or CSR feature view.
pub fn sk_validate_svm_finiteness<F: SKFloat>(view: &SKDataView<'_, F>) -> Result<(), SKError> {
    match view {
        SKDataView::Dense(dense) => {
            if let Some(position) = dense.iter().position(|value| !value.is_finite()) {
                return Err(SKError::NonFiniteValue { position });
            }
            Ok(())
        }
        SKDataView::Sparse(sparse) => {
            let columns = sparse.cols();
            for (row_index, row) in sparse.outer_iterator().enumerate() {
                if let Some((&column_index, _)) = row
                    .indices()
                    .iter()
                    .zip(row.data().iter())
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(SKError::NonFiniteValue {
                        position: row_index * columns + column_index,
                    });
                }
            }
            Ok(())
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: view.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Validate training features (non-empty, finite, square if precomputed).
pub fn sk_validate_svm_training_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    kernel: SKKernel,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = sk_svm_view_dimensions(&features)?;
    if rows == 0 || columns == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    sk_validate_svm_finiteness(&features)?;
    if kernel == SKKernel::Precomputed {
        sk_precomputed_kernel_matrix(features, rows)?;
    }
    Ok((rows, columns))
}

/// Validate query features against the fitted feature count.
pub fn sk_validate_svm_query_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    expected_columns: usize,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = sk_svm_view_dimensions(&features)?;
    if rows == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if columns != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows,
            expected_columns,
            rows,
            columns,
        ));
    }
    sk_validate_svm_finiteness(&features)?;
    Ok((rows, columns))
}
