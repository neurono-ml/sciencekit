//! Tiny-data acceptance for the neighbor estimators.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::{SKKNeighborsClassifier, SKKNeighborsRegressor};

#[test]
fn acceptance_knn_tiny_single_row_classifier_returns_its_class() {
    let features = array![[1.0_f64, 2.0]];
    let labels = array![42_i64];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let predictions = model.predict_labels(features.view()).unwrap();
    let probabilities = model.predict_probabilities(features.view()).unwrap();

    assert_eq!(predictions[0], 0);
    assert_eq!(model.classes().label_of(0), Some("42"));
    assert_eq!(probabilities.shape(), &[1, 1]);
    assert!((probabilities[[0, 0]] - 1.0).abs() < 1e-12);
}

#[test]
fn acceptance_knn_tiny_single_row_regressor_returns_its_response() {
    let features = array![[3.5_f64, -1.0]];
    let responses = array![7.25_f64];
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(features.view()).unwrap();

    assert_eq!(predictions.len(), 1);
    assert!((predictions[0] - 7.25).abs() < 1e-12);
}

#[test]
fn acceptance_knn_tiny_duplicate_rows_stay_finite() {
    let features = array![[1.0_f64], [1.0], [1.0]];
    let responses = array![2.0_f64, 4.0, 6.0];
    let model = SKKNeighborsRegressor::<f64>::build()
        .weights(crate::neighbors::SKNeighborsWeights::Distance)
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(features.view()).unwrap();

    assert!(predictions.iter().all(|value| value.is_finite()));
    assert!((predictions[0] - 4.0).abs() < 1e-12);
}
