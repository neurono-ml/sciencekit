//! Sparse cosine pairwise-distance kernel.

use ndarray::{Array, Array2, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use super::super::simd_dot::row_norm;
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
            output[[i, j]] = sk_cosine_distance(left_norms[i], right_norms[j], dot);
        }
    }
    output
}

/// Cosine distances between sparse left rows and dense right rows.
pub(crate) fn sk_sparse_dense_cosine_distance_matrix<F: SKFloat>(
    left: &CsMatView<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.cols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let right_norms: Vec<F> = right.rows().into_iter().map(row_norm).collect();
    let mut output = Array::zeros((left.rows(), right.nrows()));
    for (i, left_row) in left.outer_iterator().enumerate() {
        let left_norm = sk_sparse_row_norm(left_row.data());
        for (j, right_row) in right.rows().into_iter().enumerate() {
            let dot = left_row
                .indices()
                .iter()
                .zip(left_row.data().iter())
                .fold(F::zero(), |accumulator, (&column, &value)| {
                    accumulator + value * right_row[column]
                });
            output[[i, j]] = sk_cosine_distance(left_norm, right_norms[j], dot);
        }
    }
    output
}

/// Cosine distances between dense left rows and sparse right rows.
pub(crate) fn sk_dense_sparse_cosine_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.cols(),
        "pairwise distances require matching feature dimensions"
    );
    let left_norms: Vec<F> = left.rows().into_iter().map(row_norm).collect();
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
            let right_norm = sk_sparse_row_norm(right_row.data());
            output[[i, j]] = sk_cosine_distance(left_norms[i], right_norm, dot);
        }
    }
    output
}

/// `1 − dot / (‖a‖·‖b‖)`, keeping the dense zero-norm policy (distance `1`).
fn sk_cosine_distance<F: SKFloat>(left_norm: F, right_norm: F, dot: F) -> F {
    let denominator = left_norm * right_norm;
    if denominator == F::zero() {
        F::one()
    } else {
        F::one() - dot / denominator
    }
}
