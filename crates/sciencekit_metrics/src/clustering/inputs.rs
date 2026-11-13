//! Shared validation for the clustering metric functions.

use ndarray::ArrayView1;
use sciencekit_common::{
    SKClusterLabel, SKDataView, SKError, SKFloat, sk_validate_cluster_assignments,
};

/// Dimensions `(rows, columns)` of either feature representation.
pub(super) fn sk_view_dimensions<F>(features: &SKDataView<'_, F>) -> (usize, usize) {
    match features {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => unreachable!("SKDataView only exposes dense and csr"),
    }
}

/// Validate non-empty, row-aligned and finite clustering inputs.
///
/// Returns the feature dimensions so callers validate once and reuse them.
pub(super) fn sk_validate_clustering_inputs<F: SKFloat>(
    features: &SKDataView<'_, F>,
    assignments: ArrayView1<'_, SKClusterLabel>,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (number_of_rows, number_of_features) = sk_view_dimensions(features);
    if number_of_rows == 0 || number_of_features == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    sk_validate_cluster_assignments(number_of_rows, assignments)?;
    sk_validate_finite_features(features)?;
    Ok((number_of_rows, number_of_features))
}

/// Report the first non-finite stored value.
///
/// For CSR input only the stored elements are read: an implicit zero is a real
/// value and is always finite, so the sparse matrix is never densified.
fn sk_validate_finite_features<F: SKFloat>(features: &SKDataView<'_, F>) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            for (position, value) in dense.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
            }
        }
        SKDataView::Sparse(sparse) => {
            for (position, value) in sparse.data().iter().enumerate() {
                if !value.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
            }
        }
        _ => unreachable!("SKDataView only exposes dense and csr"),
    }
    Ok(())
}
