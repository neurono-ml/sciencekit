//! Companion tests for the marker semantics (spec `simple-imputer`).

use super::marker::{sk_is_missing, sk_sparse_marks_implicit_zeros};

#[test]
fn nan_marker_treats_only_nan_as_missing() {
    assert!(sk_is_missing(f64::NAN, f64::NAN));
    assert!(
        !sk_is_missing(0.0, f64::NAN),
        "zero is observed under a NaN marker"
    );
    assert!(
        !sk_is_missing(-7.5, f64::NAN),
        "an observed value is never missing under a NaN marker"
    );
}

#[test]
fn explicit_marker_ignores_nan_cells() {
    assert!(sk_is_missing(0.0, 0.0));
    assert!(!sk_is_missing(1.0, 0.0));
    assert!(
        !sk_is_missing(f64::NAN, 0.0),
        "a non-NaN marker never matches a NaN cell"
    );
}

#[test]
fn marker_equality_is_typed_on_f32() {
    assert!(sk_is_missing(f32::NAN, f32::NAN));
    assert!(!sk_is_missing(0.5, 0.5_f32 * 2.0));
    assert!(sk_is_missing(0.5, 0.5));
}

#[test]
fn zero_marker_marks_implicit_zeros() {
    assert!(sk_sparse_marks_implicit_zeros(0.0));
}

#[test]
fn any_other_marker_leaves_implicit_zeros_observed() {
    assert!(!sk_sparse_marks_implicit_zeros(f64::NAN));
    assert!(!sk_sparse_marks_implicit_zeros(1.0));
    assert!(!sk_sparse_marks_implicit_zeros(-3.0));
}
