//! Hand-computed pairwise distances consumed by the metric foundation.

use ndarray::array;
use sciencekit_math::pairwise::{sk_chebyshev_distance_matrix, sk_nan_euclidean_distance_matrix};

/// Chebyshev distances of a hand-computed pair of rows.
#[test]
fn chebyshev_hand_computed_matrix() {
    let left = array![[0.0_f64, 0.0], [1.0, 3.0]];
    let right = array![[0.0_f64, 0.0]];
    let distance = sk_chebyshev_distance_matrix(&left.view(), &right.view());
    assert!(distance[[0, 0]].abs() < 1e-12);
    assert!((distance[[1, 0]] - 3.0).abs() < 1e-12);
}

/// NaN-aware distance scales the co-observed difference by total/present.
#[test]
fn nan_euclidean_hand_computed_distance() {
    let left = array![[1.0_f64, f64::NAN]];
    let right = array![[3.0_f64, 5.0]];
    let distance = sk_nan_euclidean_distance_matrix(&left.view(), &right.view());
    // One co-observed feature out of two, difference 2: sqrt(2² · 2 / 1).
    assert!((distance[[0, 0]] - 8.0_f64.sqrt()).abs() < 1e-12);
}
