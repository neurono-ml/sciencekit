//! Prediction without refit and concurrent prediction (spec `grid-search`,
//! requirements `Refit` and `Concurrency and determinism`).

use std::sync::Arc;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};

use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{SKGridSearchCVClassifier, SKGridSearchCVRegressor};
use crate::split::kfold::SKKFold;

fn regression_model() -> (
    crate::grid_search::SKGridSearchCVRegressorModel<f64, SKMockConstantRegressor>,
    Array2<f64>,
) {
    let features = Array2::from_shape_fn((10, 2), |(row, column)| (row + column) as f64);
    let targets = Array1::from_iter((0..10).map(|row| (row % 3) as f64));
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(1.0),
        ])
        .splitter(SKKFold::build().number_of_splits(5).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    (model, features)
}

#[test]
fn grid_concurrency_prediction_without_refit_returns_unsupported_operation() {
    let features = Array2::zeros((10, 1));
    let targets = Array1::from_iter((0..10).map(|value| value as f64));
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![SKMockConstantRegressor::new(0.0)])
        .splitter(SKKFold::build().number_of_splits(5).build().unwrap())
        .refit(false)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();

    let error = model
        .predict(features.view())
        .err()
        .expect("prediction is blocked");
    assert!(matches!(
        error,
        SKError::UnsupportedOperation {
            operation: "predict",
            ..
        }
    ));
}

#[test]
fn grid_concurrency_classifier_without_refit_blocks_both_prediction_forms() {
    let features = Array2::from_shape_fn((12, 2), |(row, column)| (row + column) as f64);
    let labels = Array1::from_iter((0..12).map(|row| i64::from(row >= 6)));
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(vec![SKMockThresholdClassifier::new(6.0)])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .refit(false)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), labels.view()).unwrap();

    assert!(matches!(
        model.predict_labels(features.view()).unwrap_err(),
        SKError::UnsupportedOperation { .. }
    ));
    assert!(matches!(
        model.predict_probabilities(features.view()).unwrap_err(),
        SKError::UnsupportedOperation { .. }
    ));
}

#[test]
fn grid_concurrency_shared_model_predicts_identically_on_every_thread() {
    let (model, features) = regression_model();
    let model = Arc::new(model);
    let expected = model.predict(features.view()).unwrap();

    std::thread::scope(|scope| {
        for _ in 0..8 {
            let model = Arc::clone(&model);
            let features = features.view();
            let expected = expected.clone();
            scope.spawn(move || {
                assert_eq!(model.predict(features).unwrap(), expected);
            });
        }
    });
}
