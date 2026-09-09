//! Squared-Euclidean and Euclidean pairwise-distance kernels.

use ndarray::{Array, Array2, ArrayView2};
use sciencekit_common::SKFloat;

use super::simd_dot::{dot_product_simd, row_norm_squared};

/// Squared Euclidean distance matrix between the rows of two matrices.
///
/// Computes `D[i, j] = ‖a_i‖² − 2 a_i·b_j + ‖b_j‖²` using the norm-expansion
/// formulation, which is symmetric and numerically stable for the diagonal.
pub fn sk_squared_euclidean_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let (left_rows, right_rows) = (left.nrows(), right.nrows());
    let left_norms: Vec<F> = left.rows().into_iter().map(row_norm_squared).collect();
    let right_norms: Vec<F> = right.rows().into_iter().map(row_norm_squared).collect();

    let mut output = Array::zeros((left_rows, right_rows));
    for i in 0..left_rows {
        for j in 0..right_rows {
            let dot = dot_product_simd(&left.row(i), &right.row(j));
            output[[i, j]] = left_norms[i] + right_norms[j] - dot - dot;
        }
    }
    output
}

/// Euclidean distance matrix: the element-wise square root of the squared form.
pub fn sk_euclidean_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    sk_squared_euclidean_distance_matrix(left, right).mapv(|value| value.sqrt())
}
