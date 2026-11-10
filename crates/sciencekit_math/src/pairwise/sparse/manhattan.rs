//! Sparse Manhattan (L1) pairwise-distance kernel.

use ndarray::{Array, Array2, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use super::row_merge::{SKMergedCoordinate, sk_merge_sorted_rows};

/// Manhattan distance matrix between the rows of two CSR matrices.
///
/// The sorted stored coordinates are merged: a shared coordinate contributes
/// `|aᵢ − bᵢ|` and a one-sided coordinate contributes the stored value's
/// absolute value, since the implicit coordinate is an observed zero.
pub fn sk_sparse_manhattan_distance_matrix<F: SKFloat>(
    left: &CsMatView<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    assert_eq!(
        left.cols(),
        right.cols(),
        "pairwise distances require matching feature dimensions"
    );
    let mut output = Array::zeros((left.rows(), right.rows()));
    for (i, left_row) in left.outer_iterator().enumerate() {
        for (j, right_row) in right.outer_iterator().enumerate() {
            output[[i, j]] = sk_merge_absolute_differences(left_row, right_row);
        }
    }
    output
}

/// The L1 distance of two sorted sparse rows, from the stored entries only.
fn sk_merge_absolute_differences<F: SKFloat>(
    left_row: sprs::CsVecView<'_, F>,
    right_row: sprs::CsVecView<'_, F>,
) -> F {
    sk_merge_sorted_rows(
        left_row.indices(),
        left_row.data(),
        right_row.indices(),
        right_row.data(),
    )
    .fold(F::zero(), |accumulator, coordinate| {
        accumulator
            + match coordinate {
                SKMergedCoordinate::Shared {
                    left_value,
                    right_value,
                    ..
                } => (left_value - right_value).abs(),
                SKMergedCoordinate::LeftOnly { left_value, .. } => left_value.abs(),
                SKMergedCoordinate::RightOnly { right_value, .. } => right_value.abs(),
            }
    })
}

/// Manhattan distances between sparse left rows and dense right rows.
///
/// Each dense row's L1 norm is computed once; every pair then only revisits the
/// sparse stored coordinates, adjusting the dense norm with
/// `|aᵢ − bᵢ| − |bᵢ|`.
pub(crate) fn sk_sparse_dense_manhattan_distance_matrix<F: SKFloat>(
    left: &CsMatView<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.cols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let right_norms: Vec<F> = right
        .rows()
        .into_iter()
        .map(|row| row.iter().fold(F::zero(), |sum, value| sum + value.abs()))
        .collect();
    let mut output = Array::zeros((left.rows(), right.nrows()));
    for (i, left_row) in left.outer_iterator().enumerate() {
        for (j, right_row) in right.rows().into_iter().enumerate() {
            let sum = left_row
                .indices()
                .iter()
                .zip(left_row.data().iter())
                .fold(right_norms[j], |accumulator, (&column, &value)| {
                    let dense_value = right_row[column];
                    accumulator - dense_value.abs() + (value - dense_value).abs()
                });
            output[[i, j]] = sum;
        }
    }
    output
}

/// Manhattan distances between dense left rows and sparse right rows.
pub(crate) fn sk_dense_sparse_manhattan_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.cols(),
        "pairwise distances require matching feature dimensions"
    );
    let left_norms: Vec<F> = left
        .rows()
        .into_iter()
        .map(|row| row.iter().fold(F::zero(), |sum, value| sum + value.abs()))
        .collect();
    let mut output = Array::zeros((left.nrows(), right.rows()));
    for (i, left_row) in left.rows().into_iter().enumerate() {
        for (j, right_row) in right.outer_iterator().enumerate() {
            let sum = right_row
                .indices()
                .iter()
                .zip(right_row.data().iter())
                .fold(left_norms[i], |accumulator, (&column, &value)| {
                    let dense_value = left_row[column];
                    accumulator - dense_value.abs() + (value - dense_value).abs()
                });
            output[[i, j]] = sum;
        }
    }
    output
}
