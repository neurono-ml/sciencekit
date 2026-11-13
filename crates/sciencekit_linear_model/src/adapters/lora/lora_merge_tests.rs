//! Companion tests for merge/unmerge and the low-rank numerical proof (task
//! 3.1).

use ndarray::{Array1, Array2, Axis, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::adapters::lora::{SKLoraAdapterBuilder, SKLoraBaseModel, SKLoraTargetModule};
use crate::linear_regression::core_implementation::SKLinearRegression;
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

fn known_adapter() -> crate::adapters::lora::SKLoraAttachedModel<f64> {
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .alpha(2.0)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    // A = [[0.5, -0.25], [1.0, 0.75]] and B = [[2.0, -1.0]].
    attached.set_factors(array![[0.5, -0.25], [1.0, 0.75]], array![[2.0, -1.0]]);
    attached
}

#[test]
fn merged_coefficients_equal_the_hand_computed_low_rank_update() {
    let attached = known_adapter();
    let merged = attached.merge().unwrap();
    let scaling = 2.0 / 2.0;

    let base_coefficients = match &attached.base() {
        SKLoraBaseModel::LinearRegression(model) => model.coefficients().clone(),
        _ => unreachable!(),
    };
    let reference = &base_coefficients
        + &attached
            .b_matrix()
            .dot(&attached.a_matrix())
            .mapv(|value| value * scaling)
            .row(0)
            .to_owned();

    let merged_coefficients = match merged.model() {
        SKLoraBaseModel::LinearRegression(model) => model.coefficients().clone(),
        _ => unreachable!(),
    };
    for (merged, reference) in merged_coefficients.iter().zip(reference.iter()) {
        assert!(
            (merged - reference).abs() < 1e-9,
            "merged {merged}, reference {reference}"
        );
    }
}

#[test]
fn merged_prediction_reproduces_the_attached_prediction_exactly() {
    let attached = known_adapter();
    let merged = attached.merge().unwrap();
    let features = array![[0.5, -0.5], [1.5, 1.0], [-1.0, 0.25], [2.0, 2.0]];

    assert_eq!(
        merged.predict(features.view()).unwrap().to_vec(),
        attached.predict(features.view()).unwrap().to_vec(),
        "merge must fold the update without changing a single prediction bit"
    );
}

#[test]
fn unmerge_restores_the_frozen_base_and_the_original_predictions() {
    let attached = known_adapter();
    let before = attached.predict(array![[0.25, 0.5]].view()).unwrap();
    let original_coefficients = match attached.base() {
        SKLoraBaseModel::LinearRegression(model) => model.coefficients().clone(),
        _ => unreachable!(),
    };

    let restored = attached.merge().unwrap().unmerge();
    let after = restored.predict(array![[0.25, 0.5]].view()).unwrap();
    assert_eq!(before.to_vec(), after.to_vec());
    match restored.base() {
        SKLoraBaseModel::LinearRegression(model) => {
            assert_eq!(model.coefficients(), &original_coefficients);
        }
        _ => unreachable!(),
    }
    assert_eq!(restored.a_matrix(), attached.a_matrix());
    assert_eq!(restored.b_matrix(), attached.b_matrix());
}

#[test]
fn the_low_rank_reference_holds_for_single_precision() {
    let features = array![[0.0_f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let targets = features.dot(&array![2.0_f32, -3.0]) + 1.0;
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut attached = SKLoraAdapterBuilder::<f32>::new()
        .rank(1)
        .alpha(1.5)
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::linear_regression(model))
        .unwrap();
    attached.set_factors(array![[0.5_f32, -0.25]], array![[2.0_f32]]);

    let scaling = 1.5_f32;
    let (base_coefficients, intercept) = match attached.base() {
        SKLoraBaseModel::LinearRegression(model) => {
            (model.coefficients().clone(), model.intercept())
        }
        _ => unreachable!(),
    };
    let update = attached
        .b_matrix()
        .dot(&attached.a_matrix())
        .mapv(|value| value * scaling);
    let merged = base_coefficients.view().insert_axis(Axis(0)).to_owned() + update;
    let expected = features.dot(&merged.row(0)) + intercept;

    let actual = attached.predict(features.view()).unwrap();
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "f32 prediction {actual} vs reference {expected}"
        );
    }
}

#[test]
fn logistic_merge_preserves_the_probabilities() {
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
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .alpha(1.0)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::logistic_regression(model))
        .unwrap();
    attached.set_factors(array![[0.4, -0.6]], array![[1.5]]);

    let query = array![[-1.5, -0.5], [0.5, 1.5], [0.0, 0.0]];
    let attached_probabilities = attached.predict_probabilities(query.view()).unwrap();
    let merged = attached.merge().unwrap();
    let merged_probabilities = merged.predict_probabilities(query.view()).unwrap();
    assert_eq!(attached_probabilities, merged_probabilities);
    assert_eq!(
        attached.predict_labels(query.view()).unwrap().to_vec(),
        merged.predict_labels(query.view()).unwrap().to_vec()
    );
}

#[test]
fn a_fitted_adapter_can_be_merged_and_unmerged() {
    let features = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
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
    let prediction = attached.predict(features.view()).unwrap();

    let merged = attached.merge().unwrap();
    assert!(merged.is_fitted());
    assert_eq!(merged.predict(features.view()).unwrap(), prediction);
    assert_eq!(
        merged.unmerge().predict(features.view()).unwrap(),
        prediction
    );
}
