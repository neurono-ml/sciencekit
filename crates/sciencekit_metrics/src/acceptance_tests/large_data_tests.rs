//! Large-data acceptance: 100 000 observations through the public functions.

use crate::accumulators::{SKClassificationAccumulator, SKRegressionAccumulator};
use crate::classification::sk_accuracy_score;
use crate::regression::{sk_mean_absolute_error, sk_mean_squared_error};

const OBSERVATIONS: usize = 100_000;

fn labels() -> (Vec<i64>, Vec<i64>) {
    let true_labels: Vec<i64> = (0..OBSERVATIONS).map(|index| (index % 3) as i64).collect();
    let predicted_labels: Vec<i64> = true_labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            if index % 10 == 0 {
                (label + 1) % 3
            } else {
                *label
            }
        })
        .collect();
    (true_labels, predicted_labels)
}

/// Every tenth prediction is wrong, so accuracy is exactly 0.9.
#[test]
fn accuracy_on_large_data_matches_the_known_pattern() {
    let (true_labels, predicted_labels) = labels();
    let accuracy = sk_accuracy_score::<f64>(&true_labels, &predicted_labels).expect("valid input");
    assert!((accuracy - 0.9).abs() < 1e-12);
}

/// The classification accumulator matches the whole-input accuracy.
#[test]
fn classification_accumulator_on_large_data() {
    let (true_labels, predicted_labels) = labels();
    let mut accumulator = SKClassificationAccumulator::new();
    for (true_label, predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        accumulator.observe(*true_label, *predicted_label);
    }
    let matrix = accumulator.confusion_matrix(None).expect("non-empty");
    let correct: usize = (0..matrix.shape().0)
        .map(|index| {
            matrix
                .get(matrix.labels()[index], matrix.labels()[index])
                .expect("resolved label")
        })
        .sum();
    let accuracy = correct as f64 / matrix.number_of_observations() as f64;
    assert!((accuracy - 0.9).abs() < 1e-12);
}

/// Every hundredth regression prediction is off by one, so MSE and MAE are 0.01.
#[test]
fn regression_errors_on_large_data() {
    let targets: Vec<f64> = (0..OBSERVATIONS)
        .map(|index| (index % 17) as f64 * 0.5)
        .collect();
    let predictions: Vec<f64> = targets
        .iter()
        .enumerate()
        .map(|(index, target)| {
            if index % 100 == 0 {
                target + 1.0
            } else {
                *target
            }
        })
        .collect();
    let targets = ndarray::Array1::from_vec(targets);
    let predictions = ndarray::Array1::from_vec(predictions);
    let mse = sk_mean_squared_error(targets.view(), predictions.view()).expect("valid input");
    let mae = sk_mean_absolute_error(targets.view(), predictions.view()).expect("valid input");
    let expected = 1_000.0 / OBSERVATIONS as f64;
    assert!((mse - expected).abs() < 1e-12);
    assert!((mae - expected).abs() < 1e-12);

    let mut accumulator = SKRegressionAccumulator::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        accumulator.observe(*target, *prediction).expect("finite");
    }
    assert!((accumulator.mean_squared_error().unwrap() - mse).abs() < 1e-12);
}
