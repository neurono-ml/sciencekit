//! Tiny-data acceptance: a single observation still produces defined metrics.

use ndarray::array;

use crate::classification::{sk_accuracy_score, sk_confusion_matrix};
use crate::regression::{sk_mean_squared_error, sk_r2_score};

/// One correct observation scores perfect accuracy and a 1×1 matrix.
#[test]
fn single_observation_classification() {
    let accuracy = sk_accuracy_score::<f64>(&[7], &[7]).expect("valid input");
    assert!((accuracy - 1.0).abs() < 1e-12);
    let matrix = sk_confusion_matrix(&[7], &[7], None).expect("valid input");
    assert_eq!(matrix.labels(), &[7]);
    assert_eq!(matrix.get(7, 7), Some(1));
}

/// One wrong observation scores zero accuracy.
#[test]
fn single_wrong_observation_scores_zero() {
    let accuracy = sk_accuracy_score::<f64>(&[7], &[8]).expect("valid input");
    assert!(accuracy.abs() < 1e-12);
}

/// One exact prediction has zero error and a perfect coefficient.
#[test]
fn single_observation_regression() {
    let targets = array![4.5_f64];
    let predictions = array![4.5_f64];
    let mse = sk_mean_squared_error(targets.view(), predictions.view()).expect("valid input");
    let r2 = sk_r2_score(targets.view(), predictions.view()).expect("valid input");
    assert!(mse.abs() < 1e-12);
    assert!((r2 - 1.0).abs() < 1e-12);
}

/// One constant target with a residual follows the finite R² policy.
#[test]
fn single_observation_residual_scores_zero_r2() {
    let targets = array![4.5_f64];
    let predictions = array![5.5_f64];
    let r2 = sk_r2_score(targets.view(), predictions.view()).expect("valid input");
    assert!(r2.abs() < 1e-12);
}
