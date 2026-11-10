//! Sparse Chebyshev (L∞) pairwise-distance kernel.

use ndarray::{Array, Array2};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

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
    let right_rows: Vec<_> = right.outer_iterator().collect();
    let mut output = Array::zeros((left.rows(), right.rows()));
    for (i, left_row) in left.outer_iterator().enumerate() {
        for (j, right_row) in right_rows.iter().enumerate() {
            output[[i, j]] = sk_merge_sorted_rows(
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
            });
        }
    }
    output
}
