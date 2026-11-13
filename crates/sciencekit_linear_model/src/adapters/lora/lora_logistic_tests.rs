//! Companion tests for logistic-regression adapter fitting (task 2.2).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKError, SKSupervisedFit};

use crate::adapters::lora::{SKLoraAdapterBuilder, SKLoraBaseModel, SKLoraTargetModule};
use crate::logistic_regression::SKLogisticRegression;

/// A nine-point grid whose labels depend on the first coordinate at the base.
fn grid() -> Array2<f64> {
    array![
        [-2.0, -2.0],
        [-2.0, 0.0],
        [-2.0, 2.0],
        [0.0, -2.0],
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, -2.0],
        [2.0, 0.0],
        [2.0, 2.0]
    ]
}

fn labels_by_first_coordinate() -> Array1<i64> {
    Array1::from_iter(
        grid()
            .rows()
            .into_iter()
            .map(|row| if row[0] >= 0.0 { 1 } else { 0 }),
    )
}

fn labels_by_second_coordinate() -> Array1<i64> {
    Array1::from_iter(
        grid()
            .rows()
            .into_iter()
            .map(|row| if row[1] >= 0.0 { 1 } else { 0 }),
    )
}

fn logistic_base() -> SKLoraBaseModel<f64> {
    let features = grid();
    let labels = labels_by_first_coordinate();
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    SKLoraBaseModel::logistic_regression(model)
}

#[test]
fn fitting_reduces_the_logistic_loss_below_the_frozen_base() {
    let features = grid();
    let targets = labels_by_second_coordinate();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .alpha(2.0)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .maximum_number_of_iterations(500)
        .learning_rate(1.0)
        .tolerance(1e-12)
        .build()
        .unwrap()
        .attach(logistic_base())
        .unwrap();

    let before = attached
        .training_loss(features.view(), targets.view())
        .unwrap();
    attached.fit(features.view(), targets.view()).unwrap();
    let after = attached
        .training_loss(features.view(), targets.view())
        .unwrap();

    assert!(
        after < before,
        "the adapter must lower the logistic loss: before {before}, after {after}"
    );
    assert!(attached.is_fitted());
}

#[test]
fn a_classifier_adapter_rejects_continuous_targets() {
    let features = grid();
    let continuous = Array1::from_elem(features.nrows(), 0.5);
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(logistic_base())
        .unwrap();

    let error = attached
        .fit(features.view(), continuous.view())
        .expect_err("a classifier needs canonical class indices");
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

#[test]
fn a_label_outside_the_class_space_is_rejected() {
    let features = grid();
    let mut targets = labels_by_second_coordinate();
    targets[0] = 7;
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(logistic_base())
        .unwrap();

    let error = attached
        .fit(features.view(), targets.view())
        .expect_err("label 7 is outside the binary class space");
    assert!(matches!(error, SKError::UnknownLabel { label: 7 }));
}

#[test]
fn the_adapter_probabilities_stay_normalized_after_fitting() {
    let features = grid();
    let targets = labels_by_second_coordinate();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .maximum_number_of_iterations(50)
        .build()
        .unwrap()
        .attach(logistic_base())
        .unwrap();
    attached.fit(features.view(), targets.view()).unwrap();

    let probabilities = attached.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.dim(), (features.nrows(), 2));
    for row in probabilities.rows() {
        assert!((row.sum() - 1.0).abs() < 1e-12);
    }
}
