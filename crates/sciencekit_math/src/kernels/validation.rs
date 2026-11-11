//! Shared input and moment helpers for the kernel functions
//! (spec `kernel-functions`).

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CompressedStorage;

/// Dimensions `(rows, columns)` of a dense or CSR feature view.
pub(crate) fn sk_kernel_view_dimensions<F>(
    view: &SKDataView<'_, F>,
) -> Result<(usize, usize), SKError> {
    match view {
        SKDataView::Dense(dense) => Ok(dense.dim()),
        SKDataView::Sparse(sparse) => {
            if sparse.storage() != CompressedStorage::CSR {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR before computing kernels",
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

/// Reject any NaN or infinity in a dense or CSR view, naming its flat position.
pub(crate) fn sk_validate_kernel_finiteness<F: SKFloat>(
    view: &SKDataView<'_, F>,
) -> Result<(), SKError> {
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

/// Validate a `(left, right)` kernel pair: supported views, matching feature
/// dimensions and finite entries on both sides.
pub(crate) fn sk_validate_kernel_pair<F: SKFloat>(
    left: &SKDataView<'_, F>,
    right: &SKDataView<'_, F>,
) -> Result<(usize, usize), SKError> {
    let (left_rows, left_columns) = sk_kernel_view_dimensions(left)?;
    let (right_rows, right_columns) = sk_kernel_view_dimensions(right)?;
    if left_columns != right_columns {
        return Err(SKError::shape_mismatch_2d(
            left_rows,
            left_columns,
            right_rows,
            right_columns,
        ));
    }
    sk_validate_kernel_finiteness(left)?;
    sk_validate_kernel_finiteness(right)?;
    Ok((left_rows, right_rows))
}

/// Population variance over all `rows * columns` logical entries of `features`.
///
/// Implicit zeros in CSR storage contribute zero to the sum and sum of squares,
/// so a dense matrix and its CSR twin yield the exact same variance.
pub fn sk_feature_population_variance<F: SKFloat>(
    features: SKDataView<'_, F>,
) -> Result<F, SKError> {
    let (rows, columns) = sk_kernel_view_dimensions(&features)?;
    if rows == 0 || columns == 0 {
        return Err(SKError::EmptyInput {
            operation: "kernel_gamma",
        });
    }
    sk_validate_kernel_finiteness(&features)?;
    let total = F::from(rows * columns).expect("element count fits scalar");
    let mean = match features {
        SKDataView::Dense(dense) => dense.iter().fold(F::zero(), |acc, &v| acc + v) / total,
        SKDataView::Sparse(sparse) => {
            let sum = sparse.outer_iterator().fold(F::zero(), |acc, row| {
                row.data().iter().fold(acc, |row_acc, &v| row_acc + v)
            });
            sum / total
        }
        _ => F::zero(),
    };
    let squared_deviations = match features {
        SKDataView::Dense(dense) => dense.iter().fold(F::zero(), |acc, &v| {
            let delta = v - mean;
            acc + delta * delta
        }),
        SKDataView::Sparse(sparse) => {
            let mut non_zeros = 0usize;
            let stored = sparse.outer_iterator().fold(F::zero(), |acc, row| {
                non_zeros += row.nnz();
                row.data().iter().fold(acc, |row_acc, &v| {
                    let delta = v - mean;
                    row_acc + delta * delta
                })
            });
            let zero_entries = F::from(rows * columns - non_zeros).expect("count fits scalar");
            stored + zero_entries * mean * mean
        }
        _ => F::zero(),
    };
    Ok((squared_deviations / total).max(F::zero()))
}
