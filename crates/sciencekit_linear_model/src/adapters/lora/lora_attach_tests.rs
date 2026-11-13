//! Companion tests for attaching an adapter to a frozen base (task 1.2).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit};

use crate::adapters::lora::{SKLoraAdapterBuilder, SKLoraBaseModel, SKLoraTargetModule};
use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::logistic_regression::SKLogisticRegression;

/// A tiny full-rank linear dataset: `y = 2·x0 − 3·x1 + 1`.
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

#[test]
fn attach_initializes_a_and_b_at_the_base_shapes() {
    let adapter = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .alpha(4.0)
        .build()
        .unwrap();
    let attached = adapter.attach(linear_base()).unwrap();

    // One head, two features, rank two.
    assert_eq!(attached.a_matrix().dim(), (2, 2));
    assert_eq!(attached.b_matrix().dim(), (1, 2));
    assert!(!attached.is_fitted());
    assert!(attached.b_matrix().iter().all(|value| *value == 0.0));
    assert!(attached.a_matrix().iter().any(|value| *value != 0.0));
}

#[test]
fn a_zero_adapter_predicts_exactly_like_the_base() {
    let (features, _) = linear_dataset();
    let base = linear_base();
    let base_predictions = match &base {
        SKLoraBaseModel::LinearRegression(model) => model.predict(features.view()).unwrap(),
        _ => unreachable!("the helper builds a linear base"),
    };
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .alpha(4.0)
        .build()
        .unwrap()
        .attach(base)
        .unwrap();
    let attached_predictions = attached.predict(features.view()).unwrap();

    assert_eq!(
        attached_predictions.to_vec(),
        base_predictions.to_vec(),
        "a zero adapter must reproduce the frozen base bit for bit"
    );
}

#[test]
fn a_zero_logistic_adapter_reproduces_the_base_probabilities() {
    let features = array![[-1.5, -0.5], [1.5, 0.5], [0.0, 0.0]];
    let base = logistic_base();
    let base_probabilities = match &base {
        SKLoraBaseModel::LogisticRegression(model) => {
            model.predict_probabilities(features.view()).unwrap()
        }
        _ => unreachable!("the helper builds a logistic base"),
    };
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .alpha(1.0)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(base)
        .unwrap();
    let attached_probabilities = attached.predict_probabilities(features.view()).unwrap();

    assert_eq!(
        attached_probabilities, base_probabilities,
        "a zero adapter must reproduce the frozen base probabilities"
    );
}

#[test]
fn an_unsupported_base_family_is_rejected() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::unsupported_family("decision_tree"))
        .expect_err("trees have no adaptable coefficient block");

    match error {
        SKError::UnsupportedOperation { operation, reason } => {
            assert_eq!(operation, "attach");
            assert!(reason.contains("decision_tree"), "reason: {reason}");
        }
        other => panic!("expected UnsupportedOperation, found {other:?}"),
    }
}

#[test]
fn a_target_module_that_does_not_match_the_base_is_rejected() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(linear_base())
        .expect_err("a logistic target cannot adapt linear coefficients");

    match error {
        SKError::UnsupportedOperation { operation, .. } => assert_eq!(operation, "attach"),
        other => panic!("expected UnsupportedOperation, found {other:?}"),
    }
}

#[test]
fn a_rank_above_the_head_count_is_accepted_and_the_update_stays_low_rank() {
    // A linear regressor has one head; a rank above it is redundant but valid.
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(5)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    assert_eq!(attached.a_matrix().dim(), (5, 2));
    assert_eq!(attached.b_matrix().dim(), (1, 5));
    assert_eq!(attached.delta().dim(), (1, 2));
}

#[test]
fn a_classifier_only_adapter_rejects_regression_prediction() {
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(logistic_base())
        .unwrap();

    let error = attached
        .predict(array![[0.0, 0.0]].view())
        .expect_err("a classifier adapter has no continuous response");
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}
