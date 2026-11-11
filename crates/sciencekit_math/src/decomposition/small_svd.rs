//! The small singular-value decomposition of the projected matrix.
//!
//! The projected matrix `B = QᵀX` has `k + oversampling` rows, so the
//! decomposition only has to resolve a handful of directions. The backend SVD
//! returns the *full* square factors (`m × m` and `n × n`), which is
//! prohibitive for wide matrices, so the kernel decomposes the small Gram
//! matrix `B Bᵀ` with the backend eigendecomposition and rebuilds the right
//! singular vectors from `Bᵀ U`.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

use crate::backend::SKMathBackend;

/// The singular-value decomposition of the small projected matrix.
///
/// Returns the left singular vectors (`rows × rows`), the singular values in
/// non-increasing order and the right singular vectors
/// (`columns × rows`).
pub(crate) fn sk_small_svd<F: SKFloat>(
    backend: &dyn SKMathBackend<F>,
    matrix: &ArrayView2<'_, F>,
) -> Result<(Array2<F>, Vec<F>, Array2<F>), SKError> {
    let rows = matrix.nrows();
    let gram = matrix.dot(&matrix.t());
    let (eigenvalues, eigenvectors) = backend.eigh(gram.view())?;

    let mut left = Array2::<F>::zeros((rows, rows));
    let mut singular_values = Vec::with_capacity(rows);
    for order in 0..rows {
        // The eigenvalues arrive in ascending order; the singular values are
        // reported in non-increasing order.
        let source = rows - 1 - order;
        let eigenvalue = eigenvalues[source].max(F::zero());
        singular_values.push(eigenvalue.sqrt());
        left.column_mut(order).assign(&eigenvectors.column(source));
    }

    let mut right = matrix.t().dot(&left);
    for order in 0..rows {
        let singular_value = singular_values[order];
        if singular_value > F::zero() {
            right
                .column_mut(order)
                .mapv_inplace(|value| value / singular_value);
        } else {
            right.column_mut(order).fill(F::zero());
        }
    }

    Ok((left, singular_values, right))
}
