//! TDD tests for the NaN-aware Euclidean pairwise-distance kernel
//! (spec `pairwise-distances`).

use ndarray::{Array2, array};

use super::{sk_euclidean_distance_matrix, sk_nan_euclidean_distance_matrix};

/// Coordinates observed in both rows are the only ones compared.
#[test]
fn nan_euclidean_ignores_coordinates_missing_on_either_side() {
    let left: Array2<f64> = array![[1.0, f64::NAN, 2.0]];
    let right: Array2<f64> = array![[1.0, 5.0, f64::NAN]];
    let distance = sk_nan_euclidean_distance_matrix(&left.view(), &right.view());
    assert!(distance[[0, 0]].abs() < 1e-12);
}

/// The scale compensates for fewer co-observed features.
#[test]
fn nan_euclidean_scales_by_total_over_present() {
    let left: Array2<f64> = array![[0.0, f64::NAN, f64::NAN, f64::NAN]];
    let right: Array2<f64> = array![[1.5, 1.0, 2.0, 3.0]];
    let distance = sk_nan_euclidean_distance_matrix(&left.view(), &right.view());
    // One co-observed feature out of four: sqrt(1.5² · 4 / 1) = 3.
    assert!((distance[[0, 0]] - 3.0).abs() < 1e-12);
}

/// A pair with no co-observed feature yields NaN.
#[test]
fn nan_euclidean_without_co_observed_features_is_nan() {
    let left: Array2<f64> = array![[f64::NAN, 1.0]];
    let right: Array2<f64> = array![[2.0, f64::NAN]];
    let distance = sk_nan_euclidean_distance_matrix(&left.view(), &right.view());
    assert!(distance[[0, 0]].is_nan());
}

/// With every coordinate present the kernel equals the standard Euclidean one.
#[test]
fn nan_euclidean_with_present_rows_equals_euclidean() {
    let left: Array2<f64> = array![[0.0, 0.0], [3.0, 4.0]];
    let right: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0]];
    let nan_euclidean = sk_nan_euclidean_distance_matrix(&left.view(), &right.view());
    let euclidean = sk_euclidean_distance_matrix(&left.view(), &right.view());
    for ((index, value), expected) in nan_euclidean.indexed_iter().zip(euclidean.iter()) {
        assert!(
            (value - expected).abs() < 1e-12,
            "mismatch at {index:?}: {value} vs {expected}"
        );
    }
}

/// The matrix is symmetric.
#[test]
fn nan_euclidean_is_symmetric() {
    let points: Array2<f64> = array![
        [1.0, f64::NAN, 3.0],
        [1.0, 2.0, 3.0],
        [f64::NAN, 2.0, 4.0],
        [0.0, 0.0, 0.0]
    ];
    let distance = sk_nan_euclidean_distance_matrix(&points.view(), &points.view());
    for i in 0..points.nrows() {
        for j in 0..points.nrows() {
            let left = distance[[i, j]];
            let right = distance[[j, i]];
            assert!(
                (left.is_nan() && right.is_nan()) || (left - right).abs() < 1e-12,
                "asymmetric entry at ({i}, {j})"
            );
        }
    }
}

/// Large data matches a direct reference.
#[test]
fn nan_euclidean_large_data_matches_direct_computation() {
    let points: Array2<f64> = Array2::from_shape_fn((80, 12), |(i, j)| {
        if (i * 7 + j * 5) % 11 == 0 {
            f64::NAN
        } else {
            ((i * 13 + j * 3) % 41) as f64 * 0.25
        }
    });
    let distance = sk_nan_euclidean_distance_matrix(&points.view(), &points.view());
    assert_eq!(distance.shape(), &[80, 80]);
    for i in 0..80 {
        for j in 0..80 {
            let mut sum = 0.0_f64;
            let mut present = 0_usize;
            for k in 0..12 {
                let (left, right) = (points[[i, k]], points[[j, k]]);
                if !left.is_nan() && !right.is_nan() {
                    sum += (left - right) * (left - right);
                    present += 1;
                }
            }
            let expected = if present == 0 {
                f64::NAN
            } else {
                (sum * 12.0 / present as f64).sqrt()
            };
            let got = distance[[i, j]];
            assert!(
                (got.is_nan() && expected.is_nan()) || (got - expected).abs() < 1e-12,
                "mismatch at ({i}, {j}): {got} vs {expected}"
            );
        }
    }
}
