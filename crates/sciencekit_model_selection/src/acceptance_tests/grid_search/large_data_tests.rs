//! Large-data acceptance: a 1000-row search stays finite and completes.

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::super::fixtures::{cyclic_labels, dense_design};
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{SKGridSearchCVClassifier, SKGridSearchCVRegressor};
use crate::split::kfold::SKKFold;

const ROWS: usize = 1000;

#[test]
fn grid_acceptance_large_classifier_is_finite_across_every_candidate_and_fold() {
    let features = dense_design(ROWS, 4);
    let labels = cyclic_labels(ROWS, 2);
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(1.0),
            SKMockThresholdClassifier::new(4.0),
            SKMockThresholdClassifier::new(7.0),
            SKMockThresholdClassifier::new(10.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), labels.view()).unwrap();

    assert_eq!(model.fold_scores().dim(), (4, 4));
    assert!(model.fold_scores().iter().all(|score| score.is_finite()));
    assert_eq!(model.best_model().unwrap().training_rows(), ROWS);
}

#[test]
fn grid_acceptance_large_regressor_collects_every_fold() {
    let features = dense_design(ROWS, 4);
    let targets = Array1::from_iter((0..ROWS).map(|row| (row % 17) as f64));
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(8.0),
            SKMockConstantRegressor::new(16.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();

    assert_eq!(model.fold_scores().dim(), (3, 4));
    assert!(model.mean_scores().iter().all(|score| score.is_finite()));
    assert_eq!(model.best_model().unwrap().training_rows(), ROWS);
}
