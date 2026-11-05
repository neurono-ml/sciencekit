//! TDD tests for the Chebyshev pairwise-distance kernel (spec `pairwise-distances`).

use ndarray::{Array2, array};

use super::sk_chebyshev_distance_matrix;

/// Identical rows have zero Chebyshev distance.
#[test]
fn chebyshev_identical_rows_have_zero_distance() {
    let points: Array2<f64> = array![[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]];
    let distance = sk_chebyshev_distance_matrix(&points.view(), &points.view());
    assert_eq!(distance.diag(), array![0.0, 0.0, 0.0].view());
}

/// The largest absolute coordinate difference dominates the result.
#[test]
fn chebyshev_largest_coordinate_dominates() {
    let left: Array2<f64> = array![[0.0, 0.0, 0.0]];
    let right: Array2<f64> = array![[1.0, -3.0, 2.0]];
    let distance = sk_chebyshev_distance_matrix(&left.view(), &right.view());
    assert!((distance[[0, 0]] - 3.0).abs() < 1e-12);
}

/// Every entry is greater than or equal to zero.
#[test]
fn chebyshev_is_non_negative() {
    let points: Array2<f64> = array![[-1.0, 2.0], [3.0, -4.0], [0.5, 0.5]];
    let distance = sk_chebyshev_distance_matrix(&points.view(), &points.view());
    assert!(distance.iter().all(|&value| value >= 0.0));
}

/// The matrix is symmetric when both sides hold the same rows.
#[test]
fn chebyshev_is_symmetric() {
    let points: Array2<f64> = array![[0.0, 0.0], [1.0, 2.0], [3.0, 5.0], [-2.0, 1.0]];
    let distance = sk_chebyshev_distance_matrix(&points.view(), &points.view());
    for i in 0..points.nrows() {
        for j in 0..points.nrows() {
            assert!((distance[[i, j]] - distance[[j, i]]).abs() < 1e-12);
        }
    }
}

/// Unequal row counts are supported and every pair matches direct computation.
#[test]
fn chebyshev_matches_direct_computation_on_unequal_row_counts() {
    let left: Array2<f64> = array![[0.0, 0.0, 0.0], [3.0, 4.0, 1.0]];
    let right: Array2<f64> = array![[1.0, 1.0, 1.0], [0.0, -2.0, 5.0], [9.0, 9.0, 9.0]];
    let distance = sk_chebyshev_distance_matrix(&left.view(), &right.view());
    assert_eq!(distance.shape(), &[2, 3]);
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let expected = (0..left.ncols())
                .map(|k| (left[[i, k]] - right[[j, k]]).abs())
                .fold(0.0_f64, f64::max);
            assert!((distance[[i, j]] - expected).abs() < 1e-12);
        }
    }
}

/// Large data matches a direct reference (PRD §8.7: runs with lots of data).
#[test]
fn chebyshev_large_data_matches_direct_computation() {
    let points: Array2<f64> = Array2::from_shape_fn((120, 20), |(i, j)| {
        ((i * 31 + j * 17) % 101) as f64 * 0.5 - 25.0
    });
    let distance = sk_chebyshev_distance_matrix(&points.view(), &points.view());
    assert_eq!(distance.shape(), &[120, 120]);
    for i in 0..120 {
        for j in 0..120 {
            let expected = (0..20)
                .map(|k| (points[[i, k]] - points[[j, k]]).abs())
                .fold(0.0_f64, f64::max);
            assert!((distance[[i, j]] - expected).abs() < 1e-12);
        }
    }
}
