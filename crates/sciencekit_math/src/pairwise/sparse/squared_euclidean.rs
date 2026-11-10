//! Sparse squared-Euclidean and Euclidean pairwise-distance kernels.
//!
//! The norm expansion `‖a − b‖² = ‖a‖² + ‖b‖² − 2·Σ_shared aᵢbᵢ` only needs the
//! stored entries: the norms sum the stored squares and the dot product merges
//! the two sorted rows, so the work is proportional to the stored entries.

use ndarray::{Array, Array2, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use super::super::simd_dot::row_norm_squared;
use super::row_merge::sk_sparse_dot_product;
use super::row_statistics::sk_sparse_row_norm_squared;

/// Squared Euclidean distance matrix between the rows of two CSR matrices.
///
/// `D[i, j] = ‖aᵢ‖² + ‖bⱼ‖² − 2·aᵢ·bⱼ`, computed from the stored values only;
/// the inputs are never densified.
pub fn sk_sparse_squared_euclidean_distance_matrix<F: SKFloat>(
    left: &CsMatView<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    assert_eq!(
        left.cols(),
        right.cols(),
        "pairwise distances require matching feature dimensions"
    );
    let left_norms: Vec<F> = left
        .outer_iterator()
        .map(|row| sk_sparse_row_norm_squared(row.data()))
        .collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    let right_norms: Vec<F> = right_rows
        .iter()
        .map(|row| sk_sparse_row_norm_squared(row.data()))
        .collect();

    let mut output = Array::zeros((left.rows(), right.rows()));
    for (i, left_row) in left.outer_iterator().enumerate() {
        for (j, right_row) in right_rows.iter().enumerate() {
            let dot = sk_sparse_dot_product(
                left_row.indices(),
                left_row.data(),
                right_row.indices(),
                right_row.data(),
            );
            output[[i, j]] = left_norms[i] + right_norms[j] - dot - dot;
        }
    }
    output
}

/// Euclidean distance matrix over CSR rows: the square root of the sparse
/// squared form.
pub fn sk_sparse_euclidean_distance_matrix<F: SKFloat>(
    left: &CsMatView<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    sk_sparse_squared_euclidean_distance_matrix(left, right).mapv(|value| value.sqrt())
}

/// Squared Euclidean distances between sparse left rows and dense right rows.
pub(crate) fn sk_sparse_dense_squared_euclidean_distance_matrix<F: SKFloat>(
    left: &CsMatView<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.cols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let right_norms: Vec<F> = right.rows().into_iter().map(row_norm_squared).collect();
    let mut output = Array::zeros((left.rows(), right.nrows()));
    for (i, left_row) in left.outer_iterator().enumerate() {
        let left_norm = sk_sparse_row_norm_squared(left_row.data());
        for (j, right_row) in right.rows().into_iter().enumerate() {
            let dot = left_row
                .indices()
                .iter()
                .zip(left_row.data().iter())
                .fold(F::zero(), |accumulator, (&column, &value)| {
                    accumulator + value * right_row[column]
                });
            output[[i, j]] = left_norm + right_norms[j] - dot - dot;
        }
    }
    output
}

/// Squared Euclidean distances between dense left rows and sparse right rows.
pub(crate) fn sk_dense_sparse_squared_euclidean_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.cols(),
        "pairwise distances require matching feature dimensions"
    );
    let left_norms: Vec<F> = left.rows().into_iter().map(row_norm_squared).collect();
    let mut output = Array::zeros((left.nrows(), right.rows()));
    for (i, left_row) in left.rows().into_iter().enumerate() {
        for (j, right_row) in right.outer_iterator().enumerate() {
            let dot = right_row
                .indices()
                .iter()
                .zip(right_row.data().iter())
                .fold(F::zero(), |accumulator, (&column, &value)| {
                    accumulator + value * left_row[column]
                });
            let right_norm = sk_sparse_row_norm_squared(right_row.data());
            output[[i, j]] = left_norms[i] + right_norm - dot - dot;
        }
    }
    output
}

/// Euclidean distances between sparse left rows and dense right rows.
pub(crate) fn sk_sparse_dense_euclidean_distance_matrix<F: SKFloat>(
    left: &CsMatView<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    sk_sparse_dense_squared_euclidean_distance_matrix(left, right).mapv(|value| value.sqrt())
}

/// Euclidean distances between dense left rows and sparse right rows.
pub(crate) fn sk_dense_sparse_euclidean_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    sk_dense_sparse_squared_euclidean_distance_matrix(left, right).mapv(|value| value.sqrt())
}
