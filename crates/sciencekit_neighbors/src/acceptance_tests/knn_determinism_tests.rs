//! Deterministic-tie and parallel-equals-sequential acceptance.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::{SKKNeighborsClassifier, SKKNeighborsRegressor};

#[test]
fn acceptance_knn_equidistant_neighbors_use_the_smallest_indices() {
    // Every training row sits exactly two units from the query, so the three
    // selected rows must be indices 0, 1 and 2. Responses equal the index, so
    // the predicted mean proves which rows were chosen.
    let features = array![[2.0_f64], [2.0], [2.0], [2.0], [2.0], [2.0]];
    let responses = Array1::from_shape_fn(6, |row| row as f64);
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(array![[0.0_f64]].view()).unwrap();

    assert!((predictions[0] - 1.0).abs() < 1e-12);
}

#[test]
fn acceptance_knn_repeated_predictions_are_identical() {
    let features = Array2::from_shape_fn((300, 3), |(row, column)| {
        ((row * 7 + column * 5) % 31) as f64 / 3.0
    });
    let labels = Array1::from_shape_fn(300, |row| (row % 3) as i64);
    let model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let first = model.predict_labels(features.view()).unwrap();
    for _ in 0..5 {
        assert_eq!(model.predict_labels(features.view()).unwrap(), first);
    }
}

#[test]
fn acceptance_knn_parallel_batch_equals_row_by_row_prediction() {
    let features = Array2::from_shape_fn((1_200, 4), |(row, column)| {
        ((row * 13 + column * 29) % 71) as f64 / 6.0
    });
    let labels = Array1::from_shape_fn(1_200, |row| (row % 4) as i64);
    let classifier = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let regressor = SKKNeighborsRegressor::<f64>::build()
        .weights(crate::neighbors::SKNeighborsWeights::Distance)
        .build()
        .unwrap()
        .fit(
            features.view(),
            Array1::from_shape_fn(1_200, |row| row as f64).view(),
        )
        .unwrap();

    let batch_labels = classifier.predict_labels(features.view()).unwrap();
    let batch_predictions = regressor.predict(features.view()).unwrap();

    for row in 0..features.nrows() {
        let single = features
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(row..row + 1))
            .to_owned();
        let labels = classifier.predict_labels(single.view()).unwrap();
        let predictions = regressor.predict(single.view()).unwrap();
        assert_eq!(labels[0], batch_labels[row]);
        assert_eq!(predictions[0], batch_predictions[row]);
    }
}
