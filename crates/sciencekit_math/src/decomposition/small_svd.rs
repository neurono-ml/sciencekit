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

/// The decomposition of the small projected matrix.
pub(crate) struct SKSmallDecomposition<F: SKFloat> {
    /// The left singular vectors, with shape `rows × rows`.
    pub(crate) left: Array2<F>,
    /// The singular values, in non-increasing order.
    pub(crate) singular_values: Vec<F>,
    /// The right singular vectors, with shape `columns × rows`.
    pub(crate) right: Array2<F>,
}

/// Computes the singular-value decomposition of the small projected matrix.
pub(crate) fn sk_small_svd<F: SKFloat>(
    backend: &dyn SKMathBackend<F>,
    matrix: &ArrayView2<'_, F>,
) -> Result<SKSmallDecomposition<F>, SKError> {
    let rows = matrix.nrows();
    let gram = matrix.dot(&matrix.t());
    let (eigenvalues, eigenvectors) = backend.eigh(gram.view())?;

    let mut left = Array2::<F>::zeros((rows, rows));
    let mut singular_values = Vec::with_capacity(rows);
    // The eigenvalues arrive in ascending order; the singular values are
    // reported in non-increasing order.
    for source in (0..rows).rev() {
        let singular_value = eigenvalues[source].max(F::zero()).sqrt();
        let order = rows - 1 - source;
        singular_values.push(singular_value);
        left.column_mut(order).assign(&eigenvectors.column(source));
    }

    let mut right = matrix.t().dot(&left);
    for (order, &singular_value) in singular_values.iter().enumerate() {
        if singular_value > F::zero() {
            right
                .column_mut(order)
                .mapv_inplace(|value| value / singular_value);
        } else {
            right.column_mut(order).fill(F::zero());
        }
    }

    Ok(SKSmallDecomposition {
        left,
        singular_values,
        right,
    })
}
