//! Error-score policy (spec `grid-search`, requirement `Error score`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKTargetView};

use super::sk_evaluate_candidates;
use crate::grid_search::error_score::SKErrorScore;
use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::grid_search::scoring::SKRegressorScoring;
use crate::split::kfold::SKKFold;

/// Target `9` is in the training block of folds 0..4 and in the test block of
/// fold 4, so a candidate failing when it sees that target fails exactly four
/// folds and succeeds on the last one.
fn fixtures() -> (Array2<f64>, Array1<f64>) {
    let features = Array2::zeros((10, 1));
    let targets = Array1::from_iter((0..10).map(|value| value as f64));
    (features, targets)
}

fn failing_candidate() -> SKMockConstantRegressor {
    SKMockConstantRegressor::new(0.0).failing_when_target_present(9.0)
}

#[test]
fn grid_errors_raise_propagates_the_candidate_error() {
    let (features, targets) = fixtures();
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();
    let error = sk_evaluate_candidates(
        &[failing_candidate()],
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        None,
        SKErrorScore::Raise,
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .err()
    .expect("the failing candidate raises");
    assert!(matches!(error, SKError::NotConverged { iterations: 7 }));
}

#[test]
fn grid_errors_numeric_score_is_recorded_for_the_failing_fold() {
    let (features, targets) = fixtures();
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();
    let results = sk_evaluate_candidates(
        &[failing_candidate()],
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        None,
        SKErrorScore::value(-999.0),
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();

    let scores = results.fold_scores();
    assert_eq!(scores[[0, 0]], -999.0);
    assert_eq!(scores[[0, 1]], -999.0);
    assert_eq!(scores[[0, 2]], -999.0);
    assert_eq!(scores[[0, 3]], -999.0);
    assert!((scores[[0, 4]] - (-72.5)).abs() < 1e-12);
}

#[test]
fn grid_errors_numeric_score_keeps_a_healthy_candidate_winning() {
    let (features, targets) = fixtures();
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();
    let results = sk_evaluate_candidates(
        &[failing_candidate(), SKMockConstantRegressor::new(8.0)],
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        None,
        SKErrorScore::value(-999.0),
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();

    assert_eq!(results.best_index(), 1);
    let expected_healthy =
        -((64.0 + 49.0 + 36.0 + 25.0 + 16.0 + 9.0 + 4.0 + 1.0 + 0.0 + 1.0) / 10.0);
    assert!((results.best_score() - expected_healthy).abs() < 1e-12);
}
