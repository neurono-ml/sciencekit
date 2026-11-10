//! Sparse squared-Euclidean and Euclidean pairwise-distance kernels.
//!
//! The norm expansion `‖a − b‖² = ‖a‖² + ‖b‖² − 2·Σ_shared aᵢbᵢ` only needs the
//! stored entries: the norms sum the stored squares and the dot_product product merges
//! the two sorted rows, so the work is proportional to the stored entries.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use super::parallel::sk_fill_distance_matrix;
use super::row_merge::sk_sparse_dot_product;
use super::row_statistics::sk_sparse_row_norm_squared;
use crate::pairwise::simd_dot::row_norm_squared;

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
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    let left_norms: Vec<F> = left_rows
        .iter()
        .map(|row| sk_sparse_row_norm_squared(row.data()))
        .collect();
    let right_norms: Vec<F> = right_rows
        .iter()
        .map(|row| sk_sparse_row_norm_squared(row.data()))
        .collect();

    sk_fill_distance_matrix(left.rows(), right.rows(), |i, mut output_row| {
        let left_row = left_rows[i];
        for (j, right_row) in right_rows.iter().enumerate() {
            let dot_product = sk_sparse_dot_product(
                left_row.indices(),
                left_row.data(),
                right_row.indices(),
                right_row.data(),
            );
            output_row[j] = left_norms[i] + right_norms[j] - dot_product - dot_product;
        }
    })
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
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.rows().into_iter().collect();
    let right_norms: Vec<F> = right_rows
        .iter()
        .map(|row| row_norm_squared(*row))
        .collect();

    sk_fill_distance_matrix(left.rows(), right.nrows(), |i, mut output_row| {
        let left_row = left_rows[i];
        let left_norm = sk_sparse_row_norm_squared(left_row.data());
        for (j, right_row) in right_rows.iter().enumerate() {
            let dot_product = left_row
                .indices()
                .iter()
                .zip(left_row.data().iter())
                .fold(F::zero(), |accumulator, (&column, &value)| {
                    accumulator + value * right_row[column]
                });
            output_row[j] = left_norm + right_norms[j] - dot_product - dot_product;
        }
    })
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
    let left_rows: Vec<_> = left.rows().into_iter().collect();
    let left_norms: Vec<F> = left_rows.iter().map(|row| row_norm_squared(*row)).collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    let right_norms: Vec<F> = right_rows
        .iter()
        .map(|row| sk_sparse_row_norm_squared(row.data()))
        .collect();

    sk_fill_distance_matrix(left.nrows(), right.rows(), |i, mut output_row| {
        let left_row = left_rows[i];
        for (j, right_row) in right_rows.iter().enumerate() {
            let dot_product = right_row
                .indices()
                .iter()
                .zip(right_row.data().iter())
                .fold(F::zero(), |accumulator, (&column, &value)| {
                    accumulator + value * left_row[column]
                });
            output_row[j] = left_norms[i] + right_norms[j] - dot_product - dot_product;
        }
    })
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
