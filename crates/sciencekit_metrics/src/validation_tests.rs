//! TDD tests for the shared non-finite guard (spec `evaluation-metrics`).

use ndarray::array;
use sciencekit_common::SKError;

use crate::validation::sk_validate_finite_values;

/// A NaN reports its zero-based position.
#[test]
fn nan_reports_its_position() {
    let values = array![1.0_f64, 2.0, f64::NAN];
    let error = sk_validate_finite_values(&values.view()).expect_err("NaN must be rejected");
    match error {
        SKError::NonFiniteValue { position } => assert_eq!(position, 2),
        other => panic!("expected non-finite, got {other:?}"),
    }
}

/// An infinity is rejected as a non-finite value.
#[test]
fn infinity_is_rejected() {
    let values = array![f64::INFINITY, 0.0];
    let error = sk_validate_finite_values(&values.view()).expect_err("infinity must be rejected");
    assert!(matches!(error, SKError::NonFiniteValue { position: 0 }));
}

/// Finite values pass the guard.
#[test]
fn finite_values_are_accepted() {
    let values = array![-1.5_f64, 0.0, 3.25];
    assert!(sk_validate_finite_values(&values.view()).is_ok());
}
