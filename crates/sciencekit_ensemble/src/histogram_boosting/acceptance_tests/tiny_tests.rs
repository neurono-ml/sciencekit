//! Tiny-design acceptance: few rows, categorical features and missing values.

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::histogram_boosting::{
    SKHistGradientBoostingClassifier, SKHistGradientBoostingRegressor,
};

#[test]
fn hist_acceptance_tiny_binary_and_regression_fit() {
    let features = Array2::from_shape_vec(
        (8, 2),
        vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0],
    )
    .expect("data");
    let labels = ndarray::Array1::from_shape_fn(8, |row| (row % 2) as i64);
    let classifier = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(5)
        .minimum_samples_leaf(1)
        .maximum_leaf_nodes(3)
        .maximum_bins(4)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    assert_eq!(
        classifier.predict_labels(features.view()).expect("labels").dim(),
        8
    );
    let responses = ndarray::Array1::from_shape_vec((8,), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .expect("targets");
    let regressor = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(5)
        .minimum_samples_leaf(1)
        .maximum_leaf_nodes(4)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), responses.view())
        .expect("fit");
    assert_eq!(regressor.predict(features.view()).expect("predict").dim(), 8);
}

#[test]
fn hist_acceptance_categorical_and_missing_values() {
    let features = Array2::from_shape_vec(
        (10, 2),
        vec![
            10.0, 0.0, 20.0, 1.0, 10.0, f64::NAN, 30.0, 3.0, 20.0, 4.0, 10.0, 5.0, 30.0, 6.0,
            20.0, 7.0, 10.0, 8.0, 30.0, 9.0,
        ],
    )
    .expect("data");
    let responses = ndarray::Array1::from_shape_fn(10, |row| (row % 3) as f64);
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .categorical_features(vec![0])
        .number_of_estimators(6)
        .minimum_samples_leaf(1)
        .maximum_leaf_nodes(4)
        .maximum_bins(4)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), responses.view())
        .expect("fit");
    assert_eq!(model.categories(0).expect("categories"), &[10, 20, 30]);
    let predictions = model.predict(features.view()).expect("predict");
    assert!(predictions.iter().all(|value| value.is_finite()));
    let unknown = Array2::from_shape_vec((2, 2), vec![99.0, 1.0, 99.0, f64::NAN]).expect("query");
    let unknown_predictions = model.predict(unknown.view()).expect("predict");
    assert!(unknown_predictions.iter().all(|value| value.is_finite()));
}
