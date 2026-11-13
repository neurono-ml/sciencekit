//! Companion tests for linear-regression adapter fitting (task 2.1).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKRegressorPredictor, SKSupervisedFit};

use crate::adapters::lora::{SKLoraAdapterBuilder, SKLoraBaseModel};
use crate::linear_regression::core_implementation::SKLinearRegression;

/// The base relationship: `y = 2·x0 − 3·x1 + 1`.
fn base_dataset() -> (Array2<f64>, Array1<f64>) {
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0],
        [0.5, 0.5],
        [-0.5, -0.5]
    ];
    let targets = features.dot(&array![2.0, -3.0]) + 1.0;
    (features, targets)
}

fn linear_base() -> SKLoraBaseModel<f64> {
    let (features, targets) = base_dataset();
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    SKLoraBaseModel::linear_regression(model)
}

/// Data consistent with a *shifted* linear relationship the frozen base misses.
fn shifted_dataset() -> (Array2<f64>, Array1<f64>) {
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0],
        [0.5, 0.5],
        [-0.5, -0.5]
    ];
    let targets = features.dot(&array![2.7, -2.6]) + 1.0;
    (features, targets)
}

#[test]
fn fitting_reduces_the_training_loss_below_the_frozen_base() {
    let (features, targets) = shifted_dataset();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .alpha(2.0)
        .maximum_number_of_iterations(500)
        .tolerance(1e-12)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();

    let before = attached
        .training_loss(features.view(), targets.view())
        .unwrap();
    attached.fit(features.view(), targets.view()).unwrap();
    let after = attached
        .training_loss(features.view(), targets.view())
        .unwrap();

    assert!(
        after < before * 0.5,
        "the adapter must explain the shift: before {before}, after {after}"
    );
    assert!(attached.is_fitted());
}

#[test]
fn a_zero_adapter_loss_is_the_base_mean_squared_error() {
    let (features, targets) = shifted_dataset();
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let loss = attached
        .training_loss(features.view(), targets.view())
        .unwrap();
    let prediction = attached.predict(features.view()).unwrap();
    let expected: f64 = (&prediction - &targets)
        .mapv(|value| value * value)
        .mean()
        .unwrap();
    assert!((loss - expected).abs() < 1e-12);
}

#[test]
fn fitting_rejects_a_feature_count_that_disagrees_with_the_base() {
    let (features, _) = shifted_dataset();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let wrong = array![1.0, 2.0, 3.0];
    let error = attached
        .fit(array![[1.0, 2.0, 3.0]].view(), wrong.view())
        .expect_err("three features cannot fit a two-feature base");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
    let _ = features;
}

#[test]
fn fitting_rejects_a_target_length_that_disagrees_with_the_rows() {
    let (features, _) = shifted_dataset();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let short = Array1::from_elem(features.nrows() - 1, 0.0);
    let error = attached
        .fit(features.view(), short.view())
        .expect_err("each row needs one target");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn fitting_rejects_a_non_finite_target() {
    let (features, mut targets) = shifted_dataset();
    targets[2] = f64::NAN;
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let error = attached
        .fit(features.view(), targets.view())
        .expect_err("NaN targets cannot train an adapter");
    assert!(matches!(error, SKError::NonFiniteValue { .. }));
}
