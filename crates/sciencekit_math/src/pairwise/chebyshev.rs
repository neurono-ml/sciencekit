//! Chebyshev (L∞) pairwise-distance kernel.

use ndarray::{Array, Array2, ArrayView1, ArrayView2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::SKFloat;

/// Pair count below which the row-parallel path is not worth its overhead.
const SK_CHEBYSHEV_PARALLEL_GRAIN: usize = 1024;

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
    let (left_rows, right_rows) = (left.nrows(), right.nrows());
    let mut output = Array::zeros((left_rows, right_rows));
    if left_rows * right_rows < SK_CHEBYSHEV_PARALLEL_GRAIN {
        for (i, left_row) in left.rows().into_iter().enumerate() {
            for (j, right_row) in right.rows().into_iter().enumerate() {
                output[[i, j]] = sk_chebyshev_distance(&left_row, &right_row);
            }
        }
        return output;
    }
    let right_row_views: Vec<ArrayView1<'_, F>> = right.rows().into_iter().collect();
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut output_row)| {
            let left_row = left.row(i);
            for (j, right_row) in right_row_views.iter().enumerate() {
                output_row[j] = sk_chebyshev_distance(&left_row, right_row);
            }
        });
    output
}

/// The Chebyshev distance between two rows.
pub(crate) fn sk_chebyshev_distance<F: SKFloat>(left: &ArrayView1<F>, right: &ArrayView1<F>) -> F {
    let mut maximum = F::zero();
    azip!((left_value in left, right_value in right) {
        maximum = maximum.max((*left_value - *right_value).abs());
    });
    maximum
}
