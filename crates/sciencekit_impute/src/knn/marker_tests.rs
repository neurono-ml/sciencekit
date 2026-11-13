//! Companion tests for the dense missingness mask
//! (spec `nearest-neighbor-imputation`).

use ndarray::array;

use crate::knn::marker::sk_dense_missing_mask;
use crate::missingness::sk_is_missing;

#[test]
fn knn_marker_nan_marker_matches_only_nan_cells() {
    let features = array![[1.0_f64, f64::NAN], [f64::NAN, 4.0]];
    let mask = sk_dense_missing_mask(&features.view(), f64::NAN);
    assert_eq!(mask, array![[false, true], [true, false]]);
}

#[test]
fn knn_marker_scalar_marker_matches_by_equality() {
    let features = array![[-999.0_f64, 2.0], [3.0, -999.0]];
    let mask = sk_dense_missing_mask(&features.view(), -999.0);
    assert_eq!(mask, array![[true, false], [false, true]]);
}

#[test]
fn knn_marker_nan_is_observed_under_a_scalar_marker() {
    // Under a non-NaN marker a NaN cell is an observed value, never missing.
    let features = array![[f64::NAN, -999.0]];
    let mask = sk_dense_missing_mask(&features.view(), -999.0);
    assert_eq!(mask, array![[false, true]]);
}

#[test]
fn knn_marker_predicate_is_the_shared_one() {
    assert!(sk_is_missing(f64::NAN, f64::NAN));
    assert!(!sk_is_missing(0.0_f64, f64::NAN));
    assert!(sk_is_missing(-999.0_f64, -999.0));
    assert!(!sk_is_missing(f64::NAN, -999.0));
}

#[test]
fn knn_marker_mask_keeps_the_input_shape() {
    let features = ndarray::Array2::<f64>::zeros((3, 2));
    let mask = sk_dense_missing_mask(&features.view(), f64::NAN);
    assert_eq!(mask.dim(), (3, 2));
}
