//! Cosine pairwise-distance kernel.

use ndarray::{Array, Array2, ArrayView2};
use sciencekit_common::SKFloat;

use super::simd_dot::{dot_product_simd, row_norm};

/// Cosine distance matrix: `1 − cos θ` over row vectors.
///
/// Identical rows give `0`, orthogonal unit rows give `1`. Zero-norm rows yield
/// a cosine of `0` (distance `1`), matching scikit-learn's behaviour.
pub fn sk_cosine_distance_matrix<F: SKFloat>(
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
        let left_norm = row_norm(left_row);
        for (j, right_row) in right.rows().into_iter().enumerate() {
            let right_norm = row_norm(right_row);
            let dot = dot_product_simd(&left_row, &right_row);
            let denominator = left_norm * right_norm;
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
