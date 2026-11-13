//! Metric-production acceptance through `sciencekit_metrics` (task 6.1).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::sk_r2_score;

use crate::histogram_boosting::{
    SKHistGradientBoostingClassifier, SKHistGradientBoostingRegressor,
};

#[test]
fn hist_acceptance_produces_classification_and_regression_metrics() {
    let rows = 400usize;
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| {
        (row % 17) as f64 + column as f64 * 0.25
    });
    let labels =
        ndarray::Array1::from_shape_fn(rows, |row| if features[[row, 0]] > 8.0 { 1 } else { 0 });
    let classifier = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(10)
        .learning_rate(0.3)
        .maximum_leaf_nodes(6)
        .minimum_samples_leaf(1)
        .maximum_bins(16)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    let predicted = classifier.predict_labels(features.view()).expect("labels");
    let true_labels: Vec<i64> = labels.iter().copied().collect();
    let predicted_labels: Vec<i64> = predicted.iter().copied().collect();
    let accuracy: f64 = sk_accuracy_score(&true_labels, &predicted_labels).expect("accuracy");
    assert!(accuracy > 0.9, "accuracy = {accuracy}");

    let responses =
        ndarray::Array1::from_shape_fn(rows, |row| 1.5 * features[[row, 0]] - features[[row, 1]]);
    let regressor = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(12)
        .learning_rate(0.3)
        .maximum_leaf_nodes(8)
        .minimum_samples_leaf(1)
        .maximum_bins(32)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), responses.view())
        .expect("fit");
    let predictions = regressor.predict(features.view()).expect("predict");
    let r2: f64 = sk_r2_score(responses.view(), predictions.view()).expect("r2");
    assert!(r2 > 0.9, "r2 = {r2}");
}
