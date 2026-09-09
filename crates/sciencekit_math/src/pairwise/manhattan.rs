//! Manhattan (L1) pairwise-distance kernel.

use ndarray::{Array, Array2, ArrayView2};
use sciencekit_common::SKFloat;

/// Manhattan (L1) distance matrix: `Σ |a_k − b_k|`.
pub fn sk_manhattan_distance_matrix<F: SKFloat>(
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
            let mut sum = F::zero();
            for k in 0..left.ncols() {
                sum = sum + (left_row[k] - right_row[k]).abs();
            }
            output[[i, j]] = sum;
        }
    }
    output
}
