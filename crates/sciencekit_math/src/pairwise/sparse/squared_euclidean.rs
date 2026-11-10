//! Sparse squared-Euclidean and Euclidean pairwise-distance kernels.
//!
//! The norm expansion `‖a − b‖² = ‖a‖² + ‖b‖² − 2·Σ_shared aᵢbᵢ` only needs the
//! stored entries: the norms sum the stored squares and the dot product merges
//! the two sorted rows, so the work is proportional to the stored entries.

use ndarray::{Array, Array2};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

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
