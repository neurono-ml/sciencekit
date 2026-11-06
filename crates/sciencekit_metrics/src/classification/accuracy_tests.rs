//! TDD tests for the accuracy metric (spec `evaluation-metrics`).

use sciencekit_common::SKError;

use super::sk_accuracy_score;

/// A known example scores three of four observations correctly.
#[test]
fn accuracy_over_a_known_example() {
    let accuracy = sk_accuracy_score::<f64>(&[0, 1, 2, 0], &[0, 1, 2, 1]).expect("valid input");
    assert!((accuracy - 0.75).abs() < 1e-12);
}

/// Perfect predictions score one.
#[test]
fn perfect_predictions_score_one() {
    let accuracy = sk_accuracy_score::<f64>(&[0, 1, 1], &[0, 1, 1]).expect("valid input");
    assert!((accuracy - 1.0).abs() < 1e-12);
}

/// Disjoint labels score zero.
#[test]
fn disjoint_predictions_score_zero() {
    let accuracy = sk_accuracy_score::<f64>(&[0, 0], &[1, 1]).expect("valid input");
    assert!(accuracy.abs() < 1e-12);
}

/// Empty observations are rejected.
#[test]
fn empty_observations_are_rejected() {
    let error = sk_accuracy_score::<f64>(&[], &[]).expect_err("no observations");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// Different lengths are rejected as a shape error.
#[test]
fn length_mismatch_is_rejected() {
    let error = sk_accuracy_score::<f64>(&[0], &[0, 1]).expect_err("lengths differ");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
