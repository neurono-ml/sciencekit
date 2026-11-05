//! TDD tests for the mean error metrics (spec `evaluation-metrics`).

use ndarray::array;
use sciencekit_common::SKError;

use super::{sk_mean_absolute_error, sk_mean_squared_error, sk_root_mean_squared_error};

/// Perfect predictions have zero error.
#[test]
fn perfect_predictions_have_zero_error() {
    let targets = array![1.0_f64, 2.0, 3.0];
    let predictions = array![1.0_f64, 2.0, 3.0];
    let mse = sk_mean_squared_error(targets.view(), predictions.view()).expect("valid input");
    let mae = sk_mean_absolute_error(targets.view(), predictions.view()).expect("valid input");
    let rmse = sk_root_mean_squared_error(targets.view(), predictions.view()).expect("valid input");
    assert!(mse.abs() < 1e-12);
    assert!(mae.abs() < 1e-12);
    assert!(rmse.abs() < 1e-12);
}

/// The known example has MSE and MAE equal to one third.
#[test]
fn known_error_example() {
    let targets = array![1.0_f64, 2.0, 3.0];
    let predictions = array![1.0_f64, 2.0, 4.0];
    let mse = sk_mean_squared_error(targets.view(), predictions.view()).expect("valid input");
    let mae = sk_mean_absolute_error(targets.view(), predictions.view()).expect("valid input");
    let rmse = sk_root_mean_squared_error(targets.view(), predictions.view()).expect("valid input");
    assert!((mse - 1.0 / 3.0).abs() < 1e-12);
    assert!((mae - 1.0 / 3.0).abs() < 1e-12);
    assert!((rmse - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
}

/// Different lengths are rejected as a shape error.
#[test]
fn length_mismatch_is_rejected() {
    let targets = array![1.0_f64, 2.0];
    let predictions = array![1.0_f64];
    let error = sk_mean_squared_error(targets.view(), predictions.view()).expect_err("lengths");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Empty observations are rejected.
#[test]
fn empty_observations_are_rejected() {
    let targets = ndarray::Array1::<f64>::zeros(0);
    let predictions = ndarray::Array1::<f64>::zeros(0);
    let error = sk_mean_absolute_error(targets.view(), predictions.view()).expect_err("empty");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A non-finite prediction reports its position.
#[test]
fn non_finite_predictions_are_rejected() {
    let targets = array![1.0_f64, 2.0];
    let predictions = array![1.0_f64, f64::NAN];
    let error = sk_mean_squared_error(targets.view(), predictions.view()).expect_err("NaN");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}
