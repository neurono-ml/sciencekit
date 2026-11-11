//! Companion concurrency tests verifying `Send + Sync` bounds and lock-free
//! multi-threaded prediction sharing one fitted bagging model.

use std::sync::Arc;
use std::thread;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_linear_model::ridge::SKRidge;
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};
use sprs::TriMat;

use crate::{
    SKBaggingClassifier, SKBaggingClassifierModel, SKBaggingRegressor, SKBaggingRegressorModel,
};

fn assert_send_and_sync<T: Send + Sync>() {}

#[test]
fn bagging_estimators_and_fitted_models_are_send_and_sync() {
    assert_send_and_sync::<SKBaggingClassifier<SKDecisionTreeClassifier<f64>>>();
    assert_send_and_sync::<SKBaggingClassifier<SKLogisticRegression<f64>>>();
    assert_send_and_sync::<SKBaggingRegressor<SKDecisionTreeRegressor<f64>>>();
    assert_send_and_sync::<SKBaggingRegressor<SKRidge<f64>>>();

    assert_send_and_sync::<
        SKBaggingClassifierModel<
            f64,
            <SKDecisionTreeClassifier<f64> as SKSupervisedFit<f64>>::Model,
        >,
    >();
    assert_send_and_sync::<
        SKBaggingRegressorModel<f64, <SKRidge<f64> as SKSupervisedFit<f64>>::Model>,
    >();
}

#[test]
fn shared_classifier_and_regressor_models_serve_eight_threads_identically() {
    let rows = 80;
    let dense = Array2::from_shape_fn((rows, 5), |(row, column)| {
        if (row + column) % 2 == 0 {
            0.0
        } else {
            let sign = if row < 40 { -1.0 } else { 1.0 };
            sign * ((column + 1) as f64) + (row as f64) * 0.02
        }
    });
    let mut triplets = TriMat::new((rows, 5));
    dense
        .indexed_iter()
        .filter(|entry| *entry.1 != 0.0)
        .for_each(|((row, column), &value)| triplets.add_triplet(row, column, value));
    let sparse = triplets.to_csr::<usize>();
    let labels = Array1::from_shape_fn(rows, |row| i64::from(row >= 40));
    let targets = Array1::from_shape_fn(rows, |row| {
        2.0 * dense[[row, 1]] - 1.5 * dense[[row, 3]] + 0.5
    });

    let classifier_base = SKLogisticRegression::<f64>::build().build().unwrap();
    let classifier_model = Arc::new(
        SKBaggingClassifier::build(classifier_base)
            .number_of_estimators(10)
            .maximum_features(SKMaximumFeatures::Count(3))
            .bootstrap_features(true)
            .random_seed(29)
            .build()
            .unwrap()
            .fit(dense.view(), labels.view())
            .unwrap(),
    );
    assert!(classifier_model.bootstrap());
    assert!(classifier_model.bootstrap_features());

    let expected_probabilities = classifier_model
        .predict_probabilities(dense.view())
        .unwrap();
    let expected_labels = classifier_model.predict_labels(sparse.view()).unwrap();

    let regressor_base = SKRidge::<f64>::build().alpha(0.1).build().unwrap();
    let regressor_model = Arc::new(
        SKBaggingRegressor::build(regressor_base)
            .number_of_estimators(10)
            .maximum_features(SKMaximumFeatures::Count(4))
            .random_seed(29)
            .build()
            .unwrap()
            .fit(dense.view(), targets.view())
            .unwrap(),
    );
    assert!(regressor_model.bootstrap());
    assert!(!regressor_model.bootstrap_features());
    let expected_responses = regressor_model.predict(dense.view()).unwrap();

    let dense_arc = Arc::new(dense);
    let sparse_arc = Arc::new(sparse);
    let handles: Vec<_> = (0..8)
        .map(|thread_index| {
            let classifier_clone = Arc::clone(&classifier_model);
            let regressor_clone = Arc::clone(&regressor_model);
            let dense_clone = Arc::clone(&dense_arc);
            let sparse_clone = Arc::clone(&sparse_arc);
            thread::spawn(move || {
                let probabilities = if thread_index % 2 == 0 {
                    classifier_clone
                        .predict_probabilities(dense_clone.view())
                        .unwrap()
                } else {
                    classifier_clone
                        .predict_probabilities(sparse_clone.view())
                        .unwrap()
                };
                let predicted_labels = classifier_clone
                    .predict_labels(sparse_clone.view())
                    .unwrap();
                let responses = if thread_index % 2 == 0 {
                    regressor_clone.predict(dense_clone.view()).unwrap()
                } else {
                    regressor_clone.predict(sparse_clone.view()).unwrap()
                };
                (probabilities, predicted_labels, responses)
            })
        })
        .collect();

    handles.into_iter().for_each(|handle| {
        let (probabilities, predicted_labels, responses) = handle.join().unwrap();
        probabilities
            .iter()
            .zip(expected_probabilities.iter())
            .for_each(|(left, right)| assert!((left - right).abs() < 1e-12));
        assert_eq!(predicted_labels, expected_labels);
        responses
            .iter()
            .zip(expected_responses.iter())
            .for_each(|(left, right)| assert!((left - right).abs() < 1e-12));
    });
}
