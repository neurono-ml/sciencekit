//! Tiny-data acceptance: two candidates, two folds, both wrappers.

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::super::fixtures::{cyclic_labels, dense_design};
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{SKGridSearchCVClassifier, SKGridSearchCVRegressor};
use crate::split::kfold::SKKFold;

#[test]
fn grid_acceptance_tiny_classifier_selects_a_refitted_winner() {
    let features = dense_design(8, 2);
    let labels = cyclic_labels(8, 2);
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(0.0),
            SKMockThresholdClassifier::new(5.0),
        ])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), labels.view()).unwrap();

    assert_eq!(model.results().number_of_candidates(), 2);
    assert_eq!(model.results().number_of_folds(), 2);
    assert!(model.best_candidate_index() < 2);
    assert_eq!(model.best_model().unwrap().training_rows(), 8);
    assert!(model.best_score() >= -1.0 && model.best_score() <= 1.0);
}

#[test]
fn grid_acceptance_tiny_regressor_selects_a_refitted_winner() {
    let features = dense_design(8, 2);
    let targets = Array1::from_iter((0..8).map(|row| row as f64));
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(3.5),
        ])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();

    assert!(model.is_refitted());
    assert_eq!(model.ranked_candidate_indices().len(), 2);
    assert!(model.mean_scores().iter().all(|score| score.is_finite()));
}
