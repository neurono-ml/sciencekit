//! Acceptance tests for the adapter surface (task 4.1): tiny and large data,
//! deterministic fits, multiclass coverage and export.

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKClassifierPredictor, SKExportable, SKImportable, SKRegressorPredictor, SKSupervisedFit,
};

use crate::adapters::lora::{
    SKLoraAdapterArtifact, SKLoraAdapterBuilder, SKLoraBaseModel, SKLoraTargetModule,
};
use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::logistic_regression::SKLogisticRegression;

#[test]
fn a_one_row_one_feature_fit_still_produces_a_finite_prediction() {
    let base_features = array![[0.0], [1.0], [2.0]];
    let base_targets = array![0.0, 2.0, 4.0];
    let base = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(base_features.view(), base_targets.view())
        .unwrap();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .alpha(1.0)
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::linear_regression(base))
        .unwrap();

    attached
        .fit(array![[1.5]].view(), array![5.0].view())
        .unwrap();
    let prediction = attached.predict(array![[3.0]].view()).unwrap();
    assert_eq!(prediction.len(), 1);
    assert!(prediction[0].is_finite());
}

#[test]
fn a_large_design_is_fitted_and_the_loss_falls() {
    let rows = 600usize;
    let columns = 12usize;
    let features = Array2::from_shape_fn((rows, columns), |(row, column)| {
        ((row * 17 + column * 13) % 23) as f64 / 23.0 - 0.5
    });
    let coefficients = Array1::from_shape_fn(columns, |column| column as f64 / 6.0 - 1.0);
    let targets = features.dot(&coefficients) + 0.25;
    let base = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(3)
        .alpha(2.0)
        .maximum_number_of_iterations(100)
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::linear_regression(base))
        .unwrap();

    let shifted = features.dot(&(coefficients.mapv(|value| value + 0.3))) + 0.25;
    let before = attached
        .training_loss(features.view(), shifted.view())
        .unwrap();
    attached.fit(features.view(), shifted.view()).unwrap();
    let after = attached
        .training_loss(features.view(), shifted.view())
        .unwrap();
    assert!(after < before);
    assert_eq!(attached.predict(features.view()).unwrap().len(), rows);
}

#[test]
fn two_fits_on_the_same_data_produce_identical_factors() {
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0]
    ];
    let targets = features.dot(&array![2.7, -2.6]) + 1.0;
    let build = || {
        let base_features = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let base_targets = base_features.dot(&array![2.0, -3.0]) + 1.0;
        let base = SKLinearRegression::build()
            .build()
            .unwrap()
            .fit(base_features.view(), base_targets.view())
            .unwrap();
        SKLoraAdapterBuilder::<f64>::new()
            .rank(2)
            .alpha(2.0)
            .maximum_number_of_iterations(30)
            .build()
            .unwrap()
            .attach(SKLoraBaseModel::linear_regression(base))
            .unwrap()
    };
    let mut first = build();
    let mut second = build();
    first.fit(features.view(), targets.view()).unwrap();
    second.fit(features.view(), targets.view()).unwrap();

    assert_eq!(first.a_matrix(), second.a_matrix());
    assert_eq!(first.b_matrix(), second.b_matrix());
}

#[test]
fn a_multiclass_logistic_adapter_runs_and_lowers_the_loss() {
    let features = array![
        [-2.0, -2.0],
        [-2.0, -1.0],
        [0.0, 0.0],
        [0.0, 0.5],
        [2.0, 2.0],
        [2.0, 1.0]
    ];
    let base_labels = array![0_i64, 0, 1, 1, 2, 2];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), base_labels.view())
        .unwrap();
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .alpha(1.0)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .maximum_number_of_iterations(30)
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::logistic_regression(model))
        .unwrap();

    let shifted_labels = array![0_i64, 1, 1, 2, 2, 0];
    let before = attached
        .training_loss(features.view(), shifted_labels.view())
        .unwrap();
    attached
        .fit(features.view(), shifted_labels.view())
        .unwrap();
    let after = attached
        .training_loss(features.view(), shifted_labels.view())
        .unwrap();

    assert!(after.is_finite());
    assert!(after <= before);
    let probabilities = attached.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.ncols(), 3);
    for row in probabilities.rows() {
        assert!((row.sum() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn a_logistic_adapter_exports_and_imports() {
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
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::logistic_regression(model))
        .unwrap();
    attached.fit(features.view(), labels.view()).unwrap();

    let state = attached.export_state().unwrap();
    let artifact = SKLoraAdapterArtifact::<f64>::import_state(&state).unwrap();
    let base = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let restored = artifact
        .attach(SKLoraBaseModel::logistic_regression(base))
        .unwrap();
    assert_eq!(
        restored.predict_probabilities(features.view()).unwrap(),
        attached.predict_probabilities(features.view()).unwrap()
    );
}
