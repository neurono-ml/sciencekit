//! TDD tests for the coefficient of determination (spec `evaluation-metrics`).

use ndarray::array;
use sciencekit_common::SKError;

use super::sk_r2_score;

/// A perfect fit scores one.
#[test]
fn perfect_fit_scores_one() {
    let targets = array![1.0_f64, 2.0, 3.0];
    let predictions = array![1.0_f64, 2.0, 3.0];
    let score = sk_r2_score(targets.view(), predictions.view()).expect("valid input");
    assert!((score - 1.0).abs() < 1e-12);
}

/// A known non-perfect fit scores one half.
#[test]
fn known_non_perfect_fit_scores_one_half() {
    let targets = array![1.0_f64, 2.0, 3.0];
    let predictions = array![1.0_f64, 2.0, 4.0];
    let score = sk_r2_score(targets.view(), predictions.view()).expect("valid input");
    assert!((score - 0.5).abs() < 1e-12);
}

/// Constant targets with exact predictions score one.
#[test]
fn constant_targets_with_exact_predictions_score_one() {
    let targets = array![2.0_f64, 2.0, 2.0];
    let predictions = array![2.0_f64, 2.0, 2.0];
    let score = sk_r2_score(targets.view(), predictions.view()).expect("valid input");
    assert!((score - 1.0).abs() < 1e-12);
}

/// Constant targets with a non-zero residual score zero.
#[test]
fn constant_targets_with_residual_score_zero() {
    let targets = array![2.0_f64, 2.0, 2.0];
    let predictions = array![2.0_f64, 2.0, 3.0];
    let score = sk_r2_score(targets.view(), predictions.view()).expect("valid input");
    assert!(score.abs() < 1e-12);
}

/// Empty observations are rejected.
#[test]
fn empty_observations_are_rejected() {
    let targets = ndarray::Array1::<f64>::zeros(0);
    let predictions = ndarray::Array1::<f64>::zeros(0);
    let error = sk_r2_score(targets.view(), predictions.view()).expect_err("empty");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}
