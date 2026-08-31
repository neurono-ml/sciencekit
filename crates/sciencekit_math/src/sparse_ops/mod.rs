//! Sparse (`sprs`) product kernels (spec `dense-sparse-products`, PRD §4.5).
//!
//! These kernels consume zero-copy views ([`CsMatView`]) and never densify the
//! sparse operand: the CSR×dense product walks the sparse structure directly,
//! while sparse×sparse delegates to sprs' SMMP product.

use ndarray::{Array, Array2, ArrayView2, azip};
use sciencekit_common::SKFloat;
use sprs::{CsMat, CsMatView};

/// Product of a CSR matrix and a dense matrix, as a dense result.
///
/// Walks each non-zero entry of the CSR operand once (never densifying it),
/// accumulating `value * dense_row` into the matching output row via [`azip!`].
pub fn sk_csr_dense_product<F: SKFloat>(csr: &CsMatView<F>, dense: &ArrayView2<F>) -> Array2<F> {
    assert_eq!(
        csr.cols(),
        dense.nrows(),
        "CSR column count must match dense row count"
    );
    let mut output = Array::zeros((csr.rows(), dense.ncols()));
    for (row_index, sparse_row) in csr.outer_iterator().enumerate() {
        let mut output_row = output.row_mut(row_index);
        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data().iter()) {
            let dense_row = dense.row(column);
            azip!((acc in &mut output_row, d in dense_row) {
                *acc = *acc + value * *d;
            });
        }
    }
    output
}

/// Product of two sparse matrices, returning a sparse result.
///
/// Delegates to sprs' sparse matrix-matrix product (SMMP) over the two views;
/// the operands are neither densified nor copied. Sparse algebra is typed on
/// `f64` (the numeric type of Lasso, linear-SVM and text-classification
/// workloads, PRD §4.5).
pub fn sk_sparse_product(left: &CsMatView<f64>, right: &CsMatView<f64>) -> CsMat<f64> {
    assert_eq!(
        left.cols(),
        right.rows(),
        "sparse product requires lhs columns == rhs rows"
    );
    left * right
}

#[cfg(test)]
mod sparse_ops_tests;
