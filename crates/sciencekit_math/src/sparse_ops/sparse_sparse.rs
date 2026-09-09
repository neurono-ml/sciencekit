//! Sparse (`sprs`) product kernels: sparse × sparse.

use sprs::{CsMat, CsMatView};

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
