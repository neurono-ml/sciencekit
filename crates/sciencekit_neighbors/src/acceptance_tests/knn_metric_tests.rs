//! Metric-production acceptance through `sciencekit_metrics`.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::sk_r2_score;

use crate::{SKKNeighborsClassifier, SKKNeighborsRegressor};

const ROWS: usize = 400;

#[test]
fn acceptance_knn_classifier_produces_an_accuracy_score() {
    let features = Array2::from_shape_fn((ROWS, 2), |(row, column)| {
        let signal = row as f64 / ROWS as f64;
        signal + column as f64 * 0.01
    });
    let labels = Array1::from_shape_fn(ROWS, |row| if row >= ROWS / 2 { 1 } else { 0 });
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let predictions = model.predict_labels(features.view()).unwrap();
    let truth: Vec<i64> = labels
        .iter()
        .map(|label| {
            model
                .classes()
                .index_of(&label.to_string())
                .expect("fit saw every label") as i64
        })
        .collect();
    let predicted: Vec<i64> = predictions.iter().copied().collect();
    let accuracy: f64 = sk_accuracy_score(&truth, &predicted).unwrap();

    assert!(accuracy > 0.98, "accuracy was {accuracy}");
}

#[test]
fn acceptance_knn_regressor_produces_an_r2_score() {
    let features = Array2::from_shape_fn((ROWS, 2), |(row, column)| {
        let signal = row as f64 / ROWS as f64;
        signal * 2.0 + column as f64 * 0.05
    });
    let responses = Array1::from_shape_fn(ROWS, |row| {
        3.0 * features[[row, 0]] - features[[row, 1]] + 1.5
    });
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(5)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(features.view()).unwrap();
    let score = sk_r2_score(responses.view(), predictions.view()).unwrap();

    assert!(score > 0.95, "R² was {score}");
}
