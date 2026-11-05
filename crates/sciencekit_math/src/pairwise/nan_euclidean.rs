//! NaN-aware Euclidean pairwise-distance kernel.

use ndarray::{Array, Array2, ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;

/// NaN-aware Euclidean distance matrix.
///
/// The distance compares only the coordinates observed in both rows and scales
/// the accumulated squared differences by `total_features / present_features`:
/// `sqrt(Σ_present (aᵢ − bᵢ)² · total_features / present_features)`. A pair with
/// no co-observed coordinate yields NaN.
pub fn sk_nan_euclidean_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let total_features = num_traits::cast::<usize, F>(left.ncols())
        .expect("feature counts fit in the scalar");
    let mut output = Array::zeros((left.nrows(), right.nrows()));
    for (i, left_row) in left.rows().into_iter().enumerate() {
        for (j, right_row) in right.rows().into_iter().enumerate() {
            output[[i, j]] = sk_nan_euclidean_distance(&left_row, &right_row, total_features);
        }
    }
    output
}

/// The NaN-aware Euclidean distance between two rows.
fn sk_nan_euclidean_distance<F: SKFloat>(
    left: &ArrayView1<F>,
    right: &ArrayView1<F>,
    total_features: F,
) -> F {
    let (sum, present) = left.iter().zip(right.iter()).fold(
        (F::zero(), 0_usize),
        |(sum, present), (left_value, right_value)| {
            if left_value.is_nan() || right_value.is_nan() {
                (sum, present)
            } else {
                let difference = *left_value - *right_value;
                (sum + difference * difference, present + 1)
            }
        },
    );
    if present == 0 {
        return F::nan();
    }
    let present_features =
        num_traits::cast::<usize, F>(present).expect("feature counts fit in the scalar");
    (sum * total_features / present_features).sqrt()
}
