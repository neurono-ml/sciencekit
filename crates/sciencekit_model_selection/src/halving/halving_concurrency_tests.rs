//! TDD tests for prediction without refit and thread-count / concurrent invariance
//! (spec `halving-grid-search`, requirements `Refit and prediction` and
//! `Validation and concurrency`).

use std::sync::Arc;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};

use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::halving::{SKHalvingGridSearchClassifier, SKHalvingGridSearchRegressor};
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;

#[test]
fn halving_concurrency_no_refit_blocks_regressor_and_classifier_predictions() {
    let features = Array2::from_shape_fn((36, 2), |(row, col)| (row + col) as f64);
    let targets = Array1::from_iter((0..36).map(|row| (row % 4) as f64));
    let regressor = SKHalvingGridSearchRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.5),
            SKMockConstantRegressor::new(1.5),
            SKMockConstantRegressor::new(2.5),
        ])
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .minimum_resource(12)
        .refit(false)
        .build()
        .unwrap();
    let reg_model = regressor.fit(features.view(), targets.view()).unwrap();
    assert!(!reg_model.is_refitted());
    assert!(reg_model.best_model().is_none());
    assert!(matches!(
        reg_model.predict(features.view()).unwrap_err(),
        SKError::UnsupportedOperation {
            operation: "predict",
            ..
        }
    ));

    let labels = Array1::from_iter((0..36).map(|row| i64::from(row >= 18)));
    let classifier = SKHalvingGridSearchClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(10.0),
            SKMockThresholdClassifier::new(18.0),
            SKMockThresholdClassifier::new(26.0),
        ])
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(3)
                .build()
                .unwrap(),
        )
        .minimum_resource(12)
        .refit(false)
        .build()
        .unwrap();
    let clf_model = classifier.fit(features.view(), labels.view()).unwrap();
    assert!(!clf_model.is_refitted());
    assert_eq!(clf_model.best_candidate_index(), 1);
    assert!(matches!(
        clf_model.predict_labels(features.view()).unwrap_err(),
        SKError::UnsupportedOperation {
            operation: "predict_labels",
            ..
        }
    ));
    assert!(matches!(
        clf_model
            .predict_probabilities(features.view())
            .unwrap_err(),
        SKError::UnsupportedOperation {
            operation: "predict_probabilities",
            ..
        }
    ));
}

#[test]
fn halving_concurrency_parallel_fits_and_thread_pools_produce_identical_traces() {
    let features = Array2::from_shape_fn((90, 3), |(row, col)| ((row * 5 + col) % 17) as f64);
    let targets = Array1::from_iter((0..90).map(|row| ((row * 3) % 9) as f64));
    let build_estimator = || {
        let candidates: Vec<SKMockConstantRegressor> = (0..9)
            .map(|index| SKMockConstantRegressor::new(index as f64))
            .collect();
        SKHalvingGridSearchRegressor::build()
            .candidates(candidates)
            .splitter(
                SKKFold::build()
                    .number_of_splits(5)
                    .shuffle(true)
                    .random_seed(3)
                    .build()
                    .unwrap(),
            )
            .factor(3)
            .minimum_resource(10)
            .random_seed(41)
            .build()
            .unwrap()
    };

    let single_thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four_thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let baseline = single_thread_pool.install(|| {
        build_estimator()
            .fit(features.view(), targets.view())
            .unwrap()
    });
    let multi_pool = four_thread_pool.install(|| {
        build_estimator()
            .fit(features.view(), targets.view())
            .unwrap()
    });
    assert_eq!(baseline.elimination_trace(), multi_pool.elimination_trace());
    assert_eq!(
        baseline.best_candidate_index(),
        multi_pool.best_candidate_index()
    );
    assert_eq!(baseline.best_score(), multi_pool.best_score());

    let shared_model = Arc::new(baseline);
    let expected_predictions = shared_model.predict(features.view()).unwrap();
    std::thread::scope(|scope| {
        (0..8).for_each(|_| {
            let model = Arc::clone(&shared_model);
            let features_view = features.view();
            let targets_view = targets.view();
            let expected = expected_predictions.clone();
            let estimator = build_estimator();
            scope.spawn(move || {
                let fitted = estimator.fit(features_view, targets_view).unwrap();
                assert_eq!(fitted.elimination_trace(), model.elimination_trace());
                assert_eq!(model.predict(features_view).unwrap(), expected);
            });
        });
    });
}
