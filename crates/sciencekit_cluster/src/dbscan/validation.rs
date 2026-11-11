//! Input validation for DBSCAN fit and prediction (spec `dbscan`).

use sciencekit_common::{SKDataView, SKError, SKFloat};

/// Dimensions `(rows, columns)` of either supported representation.
pub(crate) fn sk_view_dimensions<F>(view: &SKDataView<'_, F>) -> (usize, usize) {
    match view {
        SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// Reject any `NaN` or infinite scalar, reporting its flat row-major position.
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
            representation: features.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Validate the feature matrix supplied to `fit` or `fit_predict`.
pub(crate) fn sk_validate_fit_features<F: SKFloat>(
    features: SKDataView<'_, F>,
) -> Result<(), SKError> {
    if !matches!(features, SKDataView::Dense(_) | SKDataView::Sparse(_)) {
        return Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        });
    }
    let (rows, columns) = sk_view_dimensions(&features);
    if rows == 0 || columns == 0 {
        return Err(SKError::EmptyInput { operation: "fit" });
    }
    sk_validate_finiteness(features)
}
