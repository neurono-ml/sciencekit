//! Histogram regressor builder and prediction tests (task 4.3).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};

use super::{SKHistGradientBoostingRegressor, SKHistGradientBoostingRegressionLoss};

fn linear_data(rows: usize) -> (Array2<f64>, ndarray::Array1<f64>) {
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| {
        ((row * 7 + column * 3) % 23) as f64 / 5.0
    });
    let targets = ndarray::Array1::from_shape_fn(rows, |row| {
        2.0 * features[[row, 0]] - features[[row, 1]] + 1.0
    });
    (features, targets)
}

#[test]
fn hist_regressor_defaults_and_loss_selector() {
    let estimator = SKHistGradientBoostingRegressor::<f64>::build()
        .build()
        .expect("defaults");
    assert_eq!(estimator.number_of_estimators(), 100);
    assert_eq!(estimator.maximum_bins(), 255);
    assert_eq!(estimator.maximum_leaf_nodes(), 31);
    assert_eq!(
        estimator.loss(),
        SKHistGradientBoostingRegressionLoss::SquaredError
    );
}

#[test]
fn hist_regressor_fits_a_linear_response() {
    let (features, targets) = linear_data(200);
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(30)
        .learning_rate(0.3)
        .maximum_leaf_nodes(8)
        .minimum_samples_leaf(2)
        .maximum_bins(32)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), targets.view())
        .expect("fit");
    let predictions = model.predict(features.view()).expect("predict");
    assert_eq!(predictions.len(), 200);
    let mean_error = predictions
        .iter()
        .zip(targets.iter())
        .map(|(predicted, actual)| (predicted - actual).abs())
        .sum::<f64>()
        / 200.0;
    assert!(mean_error < 0.5, "mean_error = {mean_error}");
    assert!(model.bin_edges(0).is_some());
    assert!(model.feature_importances().iter().sum::<f64>() > 0.0);
}

#[test]
fn hist_regressor_fits_absolute_error() {
    let (features, targets) = linear_data(120);
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .loss(SKHistGradientBoostingRegressionLoss::AbsoluteError)
        .number_of_estimators(15)
        .learning_rate(0.3)
        .maximum_leaf_nodes(6)
        .minimum_samples_leaf(1)
        .maximum_bins(16)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), targets.view())
        .expect("fit");
    let predictions = model.predict(features.view()).expect("predict");
    assert!(predictions.iter().all(|value| value.is_finite()));
    assert_eq!(model.loss(), SKHistGradientBoostingRegressionLoss::AbsoluteError);
}

#[test]
fn hist_regressor_reports_regularized_stage_values() {
    let features = Array2::from_shape_fn((40, 1), |(row, _)| row as f64);
    let targets = ndarray::Array1::from_elem(40, 3.0);
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(3)
        .learning_rate(0.5)
        .maximum_leaf_nodes(2)
        .minimum_samples_leaf(1)
        .l2_regularization(2.0)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), targets.view())
        .expect("fit");
    assert_eq!(model.initial_prediction(), 3.0);
    assert!(model.stage_trees().iter().all(|tree| tree.node_count() == 1));
    let predictions = model.predict(features.view()).expect("predict");
    assert!(predictions.iter().all(|value| (*value - 3.0).abs() < 1e-12));
}
