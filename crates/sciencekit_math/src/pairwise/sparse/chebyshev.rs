//! Sparse Chebyshev (L∞) pairwise-distance kernel.

use ndarray::{Array2, ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::{CsMatView, CsVecView};

use super::parallel::sk_fill_distance_matrix;
use super::row_merge::{SKMergedCoordinate, sk_merge_sorted_rows};

/// Chebyshev distance matrix between the rows of two CSR matrices.
///
/// The sorted stored coordinates are merged and the maximum absolute difference
/// is kept: a shared coordinate contributes `|aᵢ − bᵢ|`, a one-sided coordinate
/// contributes the stored value's absolute value, and implicit zeros therefore
/// participate in the maximum.
pub fn sk_sparse_chebyshev_distance_matrix<F: SKFloat>(
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
            output_row[j] = sk_merge_maximum_difference(&left_rows[i], right_row);
        }
    })
}

/// The L∞ distance of two sorted sparse rows, from the stored entries only.
fn sk_merge_maximum_difference<F: SKFloat>(
    left_row: &CsVecView<'_, F>,
    right_row: &CsVecView<'_, F>,
) -> F {
    sk_merge_sorted_rows(
        left_row.indices(),
        left_row.data(),
        right_row.indices(),
        right_row.data(),
    )
    .fold(F::zero(), |maximum, coordinate| {
        let difference = match coordinate {
            SKMergedCoordinate::Shared {
                left_value,
                right_value,
                ..
            } => (left_value - right_value).abs(),
            SKMergedCoordinate::LeftOnly { left_value, .. } => left_value.abs(),
            SKMergedCoordinate::RightOnly { right_value, .. } => right_value.abs(),
        };
        maximum.max(difference)
    })
}

/// Chebyshev distances between sparse left rows and dense right rows.
///
/// Unlike the other sparse×dense paths this one scans every dense coordinate
/// (`O(features)` per pair): the dense maximum may sit at an index the sparse
/// row does not store.
pub(crate) fn sk_sparse_dense_chebyshev_distance_matrix<F: SKFloat>(
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
    sk_fill_distance_matrix(left.rows(), right.nrows(), |i, mut output_row| {
        for (j, right_row) in right_rows.iter().enumerate() {
            output_row[j] = sk_sparse_dense_row_distance(&left_rows[i], *right_row);
        }
    })
}

/// Chebyshev distances between dense left rows and sparse right rows.
pub(crate) fn sk_dense_sparse_chebyshev_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &CsMatView<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.cols(),
        "pairwise distances require matching feature dimensions"
    );
    let left_rows: Vec<_> = left.rows().into_iter().collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    sk_fill_distance_matrix(left.nrows(), right.rows(), |i, mut output_row| {
        for (j, right_row) in right_rows.iter().enumerate() {
            output_row[j] = sk_sparse_dense_row_distance(right_row, left_rows[i]);
        }
    })
}

/// The L∞ distance of a sparse row and a dense row, treating implicit zeros as
/// observed zeros and scanning the full dense row.
fn sk_sparse_dense_row_distance<F: SKFloat>(
    sparse_row: &CsVecView<'_, F>,
    dense_row: ArrayView1<'_, F>,
) -> F {
    let mut stored = sparse_row
        .indices()
        .iter()
        .zip(sparse_row.data().iter())
        .peekable();
    dense_row
        .iter()
        .enumerate()
        .fold(F::zero(), |maximum, (column, &dense_value)| {
            let mut sparse_value = F::zero();
            if let Some(&(&stored_column, &value)) = stored.peek() {
                if stored_column == column {
                    sparse_value = value;
                    stored.next();
                }
            }
            maximum.max((sparse_value - dense_value).abs())
        })
}
