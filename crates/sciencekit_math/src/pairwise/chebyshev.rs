//! Chebyshev (L∞) pairwise-distance kernel.

use ndarray::{Array, Array2, ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;

/// Chebyshev (L∞) distance matrix: `max_k |a_k − b_k|`.
///
/// The distance of two rows is the largest absolute coordinate difference. The
/// matrix is symmetric when both sides hold the same rows and is always
/// non-negative.
pub fn sk_chebyshev_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let mut output = Array::zeros((left.nrows(), right.nrows()));
    for (i, left_row) in left.rows().into_iter().enumerate() {
        for (j, right_row) in right.rows().into_iter().enumerate() {
            output[[i, j]] = sk_chebyshev_distance(&left_row, &right_row);
        }
    }
    output
}

/// The Chebyshev distance between two rows.
pub(crate) fn sk_chebyshev_distance<F: SKFloat>(
    left: &ArrayView1<F>,
    right: &ArrayView1<F>,
) -> F {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| (*left_value - *right_value).abs())
        .fold(F::zero(), |maximum, difference| maximum.max(difference))
}
