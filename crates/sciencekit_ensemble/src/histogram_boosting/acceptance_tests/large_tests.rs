//! Large-design acceptance: 50 000 rows, binary/multiclass and regression.

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::histogram_boosting::{
    SKHistGradientBoostingClassifier, SKHistGradientBoostingRegressor,
};

const ROWS: usize = 50_000;

fn large_features() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, 4), |(row, column)| {
        ((row * (13 + column * 2) + column * 7) % 101) as f64 / 10.0
    })
}

#[test]
fn hist_acceptance_large_binary_classification() {
    let features = large_features();
    let labels = ndarray::Array1::from_shape_fn(ROWS, |row| ((row / 7) % 2) as i64);
    let model = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(8)
        .learning_rate(0.2)
        .maximum_leaf_nodes(8)
        .minimum_samples_leaf(20)
        .maximum_bins(32)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    let predicted = model.predict_labels(features.view()).expect("labels");
    let correct = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(predicted, actual)| predicted == actual)
        .count();
    assert!(correct * 2 > ROWS, "correct = {correct}");
}

#[test]
fn hist_acceptance_large_multiclass_classification() {
    let features = large_features();
    let labels = ndarray::Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    let model = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(4)
        .learning_rate(0.2)
        .maximum_leaf_nodes(6)
        .minimum_samples_leaf(20)
        .maximum_bins(16)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    assert_eq!(
        model
            .predict_probabilities(features.view())
            .expect("probabilities")
            .dim(),
        (ROWS, 3)
    );
}

#[test]
fn hist_acceptance_large_regression() {
    let features = large_features();
    let targets = ndarray::Array1::from_shape_fn(ROWS, |row| {
        2.0 * features[[row, 0]] - features[[row, 2]]
    });
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(10)
        .learning_rate(0.2)
        .maximum_leaf_nodes(12)
        .minimum_samples_leaf(20)
        .maximum_bins(64)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), targets.view())
        .expect("fit");
    let predictions = model.predict(features.view()).expect("predict");
    assert_eq!(predictions.len(), ROWS);
    assert!(predictions.iter().all(|value| value.is_finite()));
}
