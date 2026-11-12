//! Precomputed kernel validation wrapper (spec `kernel-functions`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::validation::sk_validate_kernel_finiteness;

/// Validate a square precomputed training kernel matrix and match its row count.
///
/// Precomputed kernel matrices are dense-only; a sparse view is rejected up
/// front with [`SKError::UnsupportedRepresentation`].
pub fn sk_precomputed_kernel_matrix<F: SKFloat>(
    matrix: SKDataView<'_, F>,
    expected_rows: usize,
) -> Result<Array2<F>, SKError> {
    let dense = matrix.as_dense()?;
    let (rows, columns) = dense.dim();
    if rows != columns {
        return Err(SKError::shape_mismatch_2d(rows, rows, rows, columns));
    }
    if rows != expected_rows {
        return Err(SKError::shape_mismatch_2d(
            expected_rows,
            expected_rows,
            rows,
            columns,
        ));
    }
    sk_validate_kernel_finiteness(&matrix)?;
    Ok(dense.to_owned())
}

/// Validate a precomputed query kernel matrix `(query_rows, training_rows)`.
pub fn sk_validate_precomputed_query_kernel<F: SKFloat>(
    matrix: SKDataView<'_, F>,
    expected_columns: usize,
) -> Result<ArrayView2<'_, F>, SKError> {
    let dense = matrix.as_dense()?;
    let (rows, columns) = dense.dim();
    if columns != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows,
            expected_columns,
            rows,
            columns,
        ));
    }
    sk_validate_kernel_finiteness(&matrix)?;
    Ok(dense)
}
