//! Sparse Manhattan (L1) pairwise-distance kernel.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::{CsMatView, CsVecView};

use super::parallel::sk_fill_distance_matrix;
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
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    sk_fill_distance_matrix(left.rows(), right.rows(), |i, mut output_row| {
        for (j, right_row) in right_rows.iter().enumerate() {
            output_row[j] = sk_merge_absolute_differences(&left_rows[i], right_row);
        }
    })
}

/// The L1 distance of two sorted sparse rows, from the stored entries only.
fn sk_merge_absolute_differences<F: SKFloat>(
    left_row: &CsVecView<'_, F>,
    right_row: &CsVecView<'_, F>,
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
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.rows().into_iter().collect();
    let right_norms: Vec<F> = right_rows
        .iter()
        .map(|row| row.iter().fold(F::zero(), |sum, value| sum + value.abs()))
        .collect();
    sk_fill_distance_matrix(left.rows(), right.nrows(), |i, mut output_row| {
        let left_row = &left_rows[i];
        for (j, right_row) in right_rows.iter().enumerate() {
            output_row[j] = left_row.indices().iter().zip(left_row.data().iter()).fold(
                right_norms[j],
                |accumulator, (&column, &value)| {
                    let dense_value = right_row[column];
                    accumulator - dense_value.abs() + (value - dense_value).abs()
                },
            );
        }
    })
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
    let left_rows: Vec<_> = left.rows().into_iter().collect();
    let left_norms: Vec<F> = left_rows
        .iter()
        .map(|row| row.iter().fold(F::zero(), |sum, value| sum + value.abs()))
        .collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    sk_fill_distance_matrix(left.nrows(), right.rows(), |i, mut output_row| {
        for (j, right_row) in right_rows.iter().enumerate() {
            output_row[j] = right_row
                .indices()
                .iter()
                .zip(right_row.data().iter())
                .fold(left_norms[i], |accumulator, (&column, &value)| {
                    let dense_value = left_rows[i][column];
                    accumulator - dense_value.abs() + (value - dense_value).abs()
                });
        }
    })
}
