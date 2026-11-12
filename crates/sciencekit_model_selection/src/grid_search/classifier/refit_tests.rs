//! Refit and best-model accessors for the classifier grid search
//! (spec `grid-search`, requirement `Refit`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::SKGridSearchCVClassifier;
use crate::grid_search::mock_estimators::SKMockThresholdClassifier;
use crate::grid_search::scoring::SKClassifierScoring;
use crate::split::kfold::SKKFold;

/// Feature value equals the row index; the label is `1` from row six on, so a
/// threshold of six is the perfect candidate.
fn fixtures() -> (Array2<f64>, Array1<i64>) {
    let features = Array2::from_shape_fn((12, 2), |(row, column)| (row + column) as f64);
    let labels = Array1::from_iter((0..12).map(|row| i64::from(row >= 6)));
    (features, labels)
}

fn estimator(refit: bool) -> SKGridSearchCVClassifier<f64, SKMockThresholdClassifier> {
    SKGridSearchCVClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(2.0),
            SKMockThresholdClassifier::new(6.0),
            SKMockThresholdClassifier::new(10.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .scoring(SKClassifierScoring::Accuracy)
        .refit(refit)
        .build()
        .unwrap()
}

#[test]
fn grid_refit_produces_a_best_model_trained_on_all_rows() {
    let (features, labels) = fixtures();
    let model = estimator(true).fit(features.view(), labels.view()).unwrap();

    assert!(model.is_refitted());
    assert_eq!(model.best_candidate_index(), 1);
    assert_eq!(model.best_parameters().threshold(), 6.0);
    assert_eq!(model.best_model().unwrap().training_rows(), 12);
}

#[test]
fn grid_refit_off_exposes_results_without_a_best_model() {
    let (features, labels) = fixtures();
    let model = estimator(false)
        .fit(features.view(), labels.view())
        .unwrap();

    assert!(!model.is_refitted());
    assert!(model.best_model().is_none());
    assert_eq!(model.results().number_of_candidates(), 3);
    assert_eq!(model.best_candidate_index(), 1);
}
