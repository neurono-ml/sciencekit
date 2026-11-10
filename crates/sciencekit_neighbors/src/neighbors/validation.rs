//! Shared validation for the brute-force searcher and the estimators
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::{SKDataView, SKError, SKFloat};

/// Dimensions `(rows, columns)` of either supported representation.
pub(crate) fn sk_view_dimensions<F>(view: &SKDataView<'_, F>) -> (usize, usize) {
    match view {
        SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// Reject a non-finite value in either representation.
pub(crate) fn sk_validate_finiteness<F: SKFloat>(
    features: SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            if let Some(position) = dense.iter().position(|value| !value.is_finite()) {
                return Err(SKError::NonFiniteValue { position });
            }
            Ok(())
        }
        SKDataView::Sparse(sparse) => {
            let columns = sparse.cols();
            for (row, stored) in sparse.outer_iterator().enumerate() {
                for (column, value) in stored.iter() {
                    if !value.is_finite() {
                        return Err(SKError::NonFiniteValue {
                            position: row * columns + column,
                        });
                    }
                }
            }
            Ok(())
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Validate a neighbor count against the training-row count.
pub(crate) fn sk_validate_neighbors_count(
    neighbors_count: usize,
    training_rows: usize,
    operation: &'static str,
) -> Result<(), SKError> {
    if neighbors_count == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "nearest_neighbors_count",
            reason: format!("must be strictly positive ({operation})"),
        });
    }
    if neighbors_count > training_rows {
        return Err(SKError::InvalidHyperparameter {
            name: "nearest_neighbors_count",
            reason: format!(
                "requested {neighbors_count} neighbors from {training_rows} training rows ({operation})"
            ),
        });
    }
    Ok(())
}
