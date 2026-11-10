//! Sparse cosine pairwise-distance kernel.

use ndarray::{Array, Array2};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use super::row_merge::sk_sparse_dot_product;
use super::row_statistics::sk_sparse_row_norm;

/// Cosine distance matrix (`1 − cos θ`) between the rows of two CSR matrices.
///
/// The dot product merges the sorted stored coordinates and each norm sums the
/// stored squares. A zero-norm row keeps the dense policy: cosine `0`, distance
/// `1`.
pub fn sk_sparse_cosine_distance_matrix<F: SKFloat>(
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
        .map(|row| sk_sparse_row_norm(row.data()))
        .collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    let right_norms: Vec<F> = right_rows
        .iter()
        .map(|row| sk_sparse_row_norm(row.data()))
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
            let denominator = left_norms[i] * right_norms[j];
            let cosine = if denominator == F::zero() {
                F::zero()
            } else {
                dot / denominator
            };
            output[[i, j]] = F::one() - cosine;
        }
    }
    output
}
