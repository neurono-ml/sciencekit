//! Shared shape validation helpers for the logistic regression module
//! (spec `logistic-regression`).
//!
//! The helpers live beside [`super::core_implementation::SKLogisticRegression`]
//! but serve the model's prediction and partial-fit paths too.

use sciencekit_common::{SKDataView, SKError, SKFloat};

/// The error for a feature representation the estimators do not consume.
pub(crate) fn unsupported_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}

/// Reject empty inputs and a label count that disagrees with the row count.
pub(crate) fn sk_validate_shape(
    rows_count: usize,
    columns_count: usize,
    labels_count: usize,
) -> Result<(), SKError> {
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    if labels_count != rows_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count],
            found: vec![labels_count],
        });
    }
    Ok(())
}

/// The row and column counts of a feature view.
pub(crate) fn sk_dimensions<F>(data: &SKDataView<'_, F>) -> (usize, usize)
where
    F: SKFloat,
{
    match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}
