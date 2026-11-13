//! Companion tests for adapter prediction and frozen-base invariance (task
//! 2.3).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit};

use crate::adapters::lora::{SKLoraAdapterBuilder, SKLoraBaseModel, SKLoraTargetModule};
use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::logistic::test_datasets::to_csr;
use crate::logistic_regression::SKLogisticRegression;

fn linear_dataset() -> (Array2<f64>, Array1<f64>) {
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0]
    ];
    let targets = features.dot(&array![2.0, -3.0]) + 1.0;
    (features, targets)
}

fn linear_base() -> SKLoraBaseModel<f64> {
    let (features, targets) = linear_dataset();
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    SKLoraBaseModel::linear_regression(model)
}

fn logistic_base() -> SKLoraBaseModel<f64> {
    let features = array![
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    SKLoraBaseModel::logistic_regression(model)
}

fn fitted_regression_adapter() -> crate::adapters::lora::SKLoraAttachedModel<f64> {
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0]
    ];
    let targets = features.dot(&array![2.7, -2.6]) + 1.0;
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .alpha(2.0)
        .maximum_number_of_iterations(50)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    attached.fit(features.view(), targets.view()).unwrap();
    attached
}

#[test]
fn fitting_leaves_the_base_coefficients_and_intercept_untouched() {
    let base = linear_base();
    let (before_coefficients, before_intercept) = match &base {
        SKLoraBaseModel::LinearRegression(model) => {
            (model.coefficients().clone(), model.intercept())
        }
        _ => unreachable!(),
    };
    let features = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let targets = array![1.0, 3.7, -2.0, 0.1];
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(base)
        .unwrap();
    attached.fit(features.view(), targets.view()).unwrap();

    match attached.base() {
        SKLoraBaseModel::LinearRegression(model) => {
            assert_eq!(model.coefficients(), &before_coefficients);
            assert_eq!(model.intercept(), before_intercept);
        }
        _ => unreachable!(),
    }
}

#[test]
fn prediction_is_the_base_plus_the_scaled_low_rank_update() {
    let attached = fitted_regression_adapter();
    let features = array![[0.5, -0.5], [1.5, 1.0], [-1.0, 0.25]];

    // The explicit reference: base prediction plus X · ((alpha / rank) B A)ᵀ.
    let base_coefficients = match &attached.base() {
        SKLoraBaseModel::LinearRegression(model) => model.coefficients().clone(),
        _ => unreachable!(),
    };
    let intercept = match &attached.base() {
        SKLoraBaseModel::LinearRegression(model) => model.intercept(),
        _ => unreachable!(),
    };
    let scaling = attached.adapter().scaling();
    let delta = attached.b_matrix().dot(&attached.a_matrix()) * scaling;
    let expected = features.dot(&base_coefficients) + features.dot(&delta.row(0)) + intercept;

    let actual = attached.predict(features.view()).unwrap();
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual {actual}, expected {expected}"
        );
    }
}

#[test]
fn the_delta_matches_the_hand_computed_low_rank_product() {
    let attached = fitted_regression_adapter();
    let scaling = attached.adapter().scaling();
    let reference = attached
        .b_matrix()
        .dot(&attached.a_matrix())
        .mapv(|value| value * scaling);
    let delta = attached.delta();
    assert_eq!(delta, reference);
}

#[test]
fn dense_and_sparse_predictions_agree_without_densifying() {
    let attached = fitted_regression_adapter();
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0]
    ];
    let sparse = to_csr(&features);

    let dense_prediction = attached.predict(features.view()).unwrap();
    let sparse_prediction = attached.predict(sparse.view()).unwrap();
    for (dense, sparse) in dense_prediction.iter().zip(sparse_prediction.iter()) {
        assert!(
            (dense - sparse).abs() < 1e-12,
            "dense {dense}, sparse {sparse}"
        );
    }
}

#[test]
fn logistic_dense_and_sparse_decision_scores_agree() {
    let features = array![[-1.5, -0.5], [1.5, 0.5], [0.0, 0.0], [0.25, 1.5]];
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .alpha(1.0)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(logistic_base())
        .unwrap();
    let sparse = to_csr(&features);

    let dense = attached.decision_function(features.view()).unwrap();
    let sparse_scores = attached.decision_function(sparse.view()).unwrap();
    for (dense, sparse) in dense.iter().zip(sparse_scores.iter()) {
        assert!((dense - sparse).abs() < 1e-12);
    }
}

#[test]
fn prediction_rejects_a_feature_count_that_disagrees_with_the_base() {
    let attached = fitted_regression_adapter();
    let error = attached
        .predict(array![[1.0, 2.0, 3.0]].view())
        .expect_err("three features cannot predict a two-feature adapter");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn logistic_prediction_returns_one_label_per_row() {
    let features = array![
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0]
    ];
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(logistic_base())
        .unwrap();
    let labels = attached.predict_labels(features.view()).unwrap();
    assert_eq!(labels.to_vec(), vec![0, 0, 0, 1, 1, 1]);
}
