//! Input validation for `SKIsolationForest` fit and query operations.

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CompressedStorage;

use super::config::SKIsolationMaximumSamples;

/// Returns `(rows, columns)` for a dense or sparse view.
pub(super) fn sk_view_dimensions<F>(features: &SKDataView<'_, F>) -> (usize, usize) {
    match features {
        SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// Rejects unsupported representations (including CSC sparse views, naming CSR).
fn sk_validate_supported_view<F>(features: &SKDataView<'_, F>) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(_) => Ok(()),
        SKDataView::Sparse(sparse) => {
            if sparse.storage() != CompressedStorage::CSR {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR with to_csr",
                });
            }
            Ok(())
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Rejects any `NaN` or infinite scalar, reporting its flat row-major position.
pub(super) fn sk_validate_finiteness<F: SKFloat>(
    features: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            let columns = dense.ncols();
            let non_finite =
                dense
                    .rows()
                    .into_iter()
                    .enumerate()
                    .find_map(|(row_index, row_values)| {
                        row_values
                            .iter()
                            .position(|value| !value.is_finite())
                            .map(|column_index| row_index * columns + column_index)
                    });
            if let Some(position) = non_finite {
                return Err(SKError::NonFiniteValue { position });
            }
            Ok(())
        }
        SKDataView::Sparse(sparse) => {
            let columns = sparse.cols();
            let non_finite =
                sparse
                    .outer_iterator()
                    .enumerate()
                    .find_map(|(row_index, stored_row)| {
                        stored_row
                            .indices()
                            .iter()
                            .zip(stored_row.data().iter())
                            .find(|&(_, value)| !value.is_finite())
                            .map(|(&column_index, _)| row_index * columns + column_index)
                    });
            if let Some(position) = non_finite {
                return Err(SKError::NonFiniteValue { position });
            }
            Ok(())
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Validates training features and sample-count bounds before fitting.
pub(super) fn sk_validate_fit_features<F: SKFloat>(
    features: &SKDataView<'_, F>,
    maximum_samples: SKIsolationMaximumSamples<F>,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    sk_validate_supported_view(features)?;
    let (rows_count, columns_count) = sk_view_dimensions(features);
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if let SKIsolationMaximumSamples::Count(explicit_count) = maximum_samples {
        if explicit_count > rows_count {
            return Err(SKError::InvalidHyperparameter {
                name: "maximum_samples",
                reason: format!(
                    "explicit maximum_samples ({explicit_count}) exceeds training row count ({rows_count})"
                ),
            });
        }
    }
    sk_validate_finiteness(features)?;
    Ok((rows_count, columns_count))
}

/// Validates query features against `expected_columns` before scoring or predicting.
pub(super) fn sk_validate_query_features<F: SKFloat>(
    features: &SKDataView<'_, F>,
    expected_columns: usize,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    sk_validate_supported_view(features)?;
    let (rows_count, columns_count) = sk_view_dimensions(features);
    if rows_count == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if columns_count != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows_count,
            expected_columns,
            rows_count,
            columns_count,
        ));
    }
    sk_validate_finiteness(features)?;
    Ok((rows_count, columns_count))
}
