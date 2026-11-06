//! `f32` metrics match an `f64` reference within the documented tolerance.

use ndarray::array;

use crate::classification::sk_accuracy_score;
use crate::regression::{sk_mean_squared_error, sk_r2_score};

/// Classification accuracy in `f32` matches the `f64` result.
#[test]
fn f32_accuracy_matches_f64_reference() {
    let true_labels = [0_i64, 1, 2, 0, 1, 2, 0, 1, 2, 0];
    let predicted_labels = [0_i64, 1, 2, 0, 2, 2, 0, 1, 1, 0];
    let f32_score = sk_accuracy_score::<f32>(&true_labels, &predicted_labels).expect("valid");
    let f64_score = sk_accuracy_score::<f64>(&true_labels, &predicted_labels).expect("valid");
    assert!((f32_score as f64 - f64_score).abs() < 1e-6);
}

/// Regression metrics in `f32` match the `f64` result.
#[test]
fn f32_regression_metrics_match_f64_reference() {
    let targets_32 = array![1.0_f32, 2.0, 3.0, 4.0];
    let predictions_32 = array![1.0_f32, 2.0, 4.0, 4.0];
    let targets_64 = array![1.0_f64, 2.0, 3.0, 4.0];
    let predictions_64 = array![1.0_f64, 2.0, 4.0, 4.0];

    let mse_32 =
        sk_mean_squared_error(targets_32.view(), predictions_32.view()).expect("valid") as f64;
    let mse_64 = sk_mean_squared_error(targets_64.view(), predictions_64.view()).expect("valid");
    assert!((mse_32 - mse_64).abs() < 1e-5);

    let r2_32 = sk_r2_score(targets_32.view(), predictions_32.view()).expect("valid") as f64;
    let r2_64 = sk_r2_score(targets_64.view(), predictions_64.view()).expect("valid");
    assert!((r2_32 - r2_64).abs() < 1e-5);
}
