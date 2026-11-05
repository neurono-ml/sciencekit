//! TDD tests for the regression streaming accumulator (spec `evaluation-metrics`).

use ndarray::array;
use sciencekit_common::SKError;

use super::SKRegressionAccumulator;
use crate::regression::{
    sk_mean_absolute_error, sk_mean_squared_error, sk_r2_score, sk_root_mean_squared_error,
};

fn observed_pairs() -> ([f64; 5], [f64; 5]) {
    ([1.0, 2.0, 3.0, 4.0, 10.0], [1.0, 2.0, 4.0, 4.0, 8.0])
}

/// Batched accumulation equals whole-input evaluation.
#[test]
fn batched_accumulation_equals_whole_input() {
    let (targets, predictions) = observed_pairs();
    let mut accumulator = SKRegressionAccumulator::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        accumulator
            .observe(*target, *prediction)
            .expect("finite values");
    }
    let targets = array![1.0_f64, 2.0, 3.0, 4.0, 10.0];
    let predictions = array![1.0_f64, 2.0, 4.0, 4.0, 8.0];
    let mse = accumulator.mean_squared_error().expect("non-empty");
    let expected = sk_mean_squared_error(targets.view(), predictions.view()).expect("valid");
    assert!((mse - expected).abs() < 1e-12);
    let rmse = accumulator.root_mean_squared_error().expect("non-empty");
    let expected = sk_root_mean_squared_error(targets.view(), predictions.view()).expect("valid");
    assert!((rmse - expected).abs() < 1e-12);
    let mae = accumulator.mean_absolute_error().expect("non-empty");
    let expected = sk_mean_absolute_error(targets.view(), predictions.view()).expect("valid");
    assert!((mae - expected).abs() < 1e-12);
    let r2 = accumulator.r2_score().expect("non-empty");
    let expected = sk_r2_score(targets.view(), predictions.view()).expect("valid");
    assert!((r2 - expected).abs() < 1e-12);
}

/// Merging two accumulators equals one pass over the concatenation.
#[test]
fn merging_two_accumulators_equals_concatenation() {
    let (targets, predictions) = observed_pairs();
    let mut left = SKRegressionAccumulator::new();
    for (target, prediction) in targets[..2].iter().zip(predictions[..2].iter()) {
        left.observe(*target, *prediction).expect("finite values");
    }
    let mut right = SKRegressionAccumulator::new();
    for (target, prediction) in targets[2..].iter().zip(predictions[2..].iter()) {
        right.observe(*target, *prediction).expect("finite values");
    }
    left.merge(&right);
    let mut whole = SKRegressionAccumulator::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        whole.observe(*target, *prediction).expect("finite values");
    }
    assert!(
        (left.mean_squared_error().unwrap() - whole.mean_squared_error().unwrap()).abs() < 1e-12
    );
    assert!((left.r2_score().unwrap() - whole.r2_score().unwrap()).abs() < 1e-12);
    assert_eq!(left.number_of_observations(), 5);
}

/// An empty accumulator is rejected by every metric finalizer.
#[test]
fn empty_accumulator_is_rejected() {
    let accumulator: SKRegressionAccumulator<f64> = SKRegressionAccumulator::new();
    assert!(matches!(
        accumulator.mean_squared_error(),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        accumulator.root_mean_squared_error(),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        accumulator.mean_absolute_error(),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        accumulator.r2_score(),
        Err(SKError::EmptyInput { .. })
    ));
}

/// A non-finite observation reports the observation index.
#[test]
fn non_finite_observation_is_rejected() {
    let mut accumulator = SKRegressionAccumulator::new();
    accumulator.observe(1.0_f64, 1.0).expect("finite");
    let error = accumulator
        .observe(f64::NAN, 1.0)
        .expect_err("NaN must be rejected");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}

/// Large-magnitude `f32` targets keep batched R² equal to whole-input R².
#[test]
fn f32_large_magnitude_r2_matches_whole_input() {
    let count = 2_000_usize;
    let targets: Vec<f32> = (0..count)
        .map(|index| 3.0e6 + (index % 7) as f32 * 0.5)
        .collect();
    let predictions: Vec<f32> = targets
        .iter()
        .enumerate()
        .map(|(index, target)| target + ((index % 5) as f32 - 2.0) * 0.1)
        .collect();
    let targets_array = ndarray::Array1::from_vec(targets.clone());
    let predictions_array = ndarray::Array1::from_vec(predictions.clone());
    let expected =
        sk_r2_score(targets_array.view(), predictions_array.view()).expect("valid input");

    let mut accumulator = SKRegressionAccumulator::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        accumulator.observe(*target, *prediction).expect("finite");
    }
    let got = accumulator.r2_score().expect("non-empty");
    assert!(
        (got - expected).abs() < 1e-3,
        "batched f32 R² {got} differs from whole-input {expected}"
    );
    assert!(
        got > 0.0 && got < 1.0,
        "must not trigger the constant policy: {got}"
    );
}
