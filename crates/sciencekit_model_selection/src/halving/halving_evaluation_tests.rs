//! TDD tests for successive-halving round evaluation over subsampled folds
//! (spec `halving-grid-search`, requirements `Successive halving loop`,
//! `Deterministic elimination` and `Resource subsampling`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKError};

use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::grid_search::{SKErrorScore, SKRegressorScoring};
use crate::split::kfold::SKKFold;

use super::SKHalvingResource;
use super::evaluation::{SKHalvingEvaluationRequest, sk_run_halving_search};

#[test]
fn halving_evaluation_runs_nine_three_one_and_selects_true_target_constant() {
    let features = Array2::from_shape_fn((90, 2), |(row, col)| (row + col) as f64);
    let targets = Array1::from_elem(90, 4.0);
    let candidates: Vec<SKMockConstantRegressor> = (0..9)
        .map(|index| SKMockConstantRegressor::new(index as f64))
        .collect();
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();

    let outcome = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 3,
            minimum_resource: 10,
            maximum_resource: None,
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: false,
            random_seed: 17,
            error_score: SKErrorScore::Raise,
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();

    assert_eq!(outcome.rounds.len(), 2);
    assert_eq!(outcome.rounds[0].resource(), 10);
    assert_eq!(outcome.rounds[0].evaluated_candidate_indices().len(), 9);
    assert_eq!(outcome.rounds[0].surviving_candidate_indices().len(), 3);
    assert_eq!(outcome.rounds[1].resource(), 30);
    assert_eq!(outcome.rounds[1].evaluated_candidate_indices().len(), 3);
    assert_eq!(outcome.rounds[1].surviving_candidate_indices().len(), 1);
    assert_eq!(outcome.best_candidate_index, 4);
    assert_eq!(outcome.best_score, 0.0);
}

#[test]
fn halving_evaluation_stops_at_maximum_resource_and_picks_highest_mean_with_tie_break() {
    let features = Array2::zeros((30, 2));
    let targets = Array1::from_elem(30, 2.0);
    // Candidates 1 and 3 tie at value 2.0 (score 0.0); candidate 1 has the smaller index.
    let candidates = vec![
        SKMockConstantRegressor::new(0.0),
        SKMockConstantRegressor::new(2.0),
        SKMockConstantRegressor::new(5.0),
        SKMockConstantRegressor::new(2.0),
        SKMockConstantRegressor::new(8.0),
        SKMockConstantRegressor::new(9.0),
    ];
    let splitter = SKKFold::build().number_of_splits(3).build().unwrap();

    let run = || {
        sk_run_halving_search(
            SKHalvingEvaluationRequest {
                candidates: &candidates,
                splitter: &splitter,
                features: features.view().try_into().unwrap(),
                targets: targets.view().try_into().unwrap(),
                groups: None,
                factor: 2,
                minimum_resource: 12,
                maximum_resource: Some(24),
                resource: SKHalvingResource::NumberOfSamples,
                aggressive_elimination: false,
                random_seed: 5,
                error_score: SKErrorScore::Raise,
            },
            &SKRegressorScoring::NegativeMeanSquaredError,
        )
        .unwrap()
    };

    let first = run();
    let second = run();
    assert_eq!(first, second);
    // 6 -> 3 at r=12, then 3 -> 2 at r=24 (maximum resource reached with 2 survivors).
    assert_eq!(first.rounds.len(), 2);
    assert_eq!(first.rounds[0].resource(), 12);
    assert_eq!(first.rounds[1].resource(), 24);
    assert_eq!(first.rounds[1].surviving_candidate_indices().len(), 2);
    assert_eq!(first.best_candidate_index, 1);
    assert_eq!(first.best_score, 0.0);
}

#[test]
fn halving_evaluation_respects_raise_and_value_error_score_policies() {
    let features = Array2::zeros((18, 1));
    let targets = Array1::from_elem(18, 3.0);
    let candidates = vec![
        SKMockConstantRegressor::new(0.0).failing_when_target_present(3.0),
        SKMockConstantRegressor::new(3.0),
    ];
    let splitter = SKKFold::build().number_of_splits(3).build().unwrap();

    let raise_error = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 2,
            minimum_resource: 9,
            maximum_resource: None,
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: false,
            random_seed: 0,
            error_score: SKErrorScore::Raise,
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap_err();
    assert!(matches!(
        raise_error,
        SKError::NotConverged { iterations: 7 }
    ));

    let recovered = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 2,
            minimum_resource: 9,
            maximum_resource: None,
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: false,
            random_seed: 0,
            error_score: SKErrorScore::Value(f64::NAN),
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();
    assert_eq!(recovered.best_candidate_index, 1);
    assert_eq!(recovered.best_score, 0.0);
}

#[test]
fn halving_evaluation_single_candidate_still_evaluates_one_round() {
    let features = Array2::zeros((12, 1));
    let targets = Array1::from_elem(12, 1.5);
    let candidates = vec![SKMockConstantRegressor::new(1.5)];
    let splitter = SKKFold::build().number_of_splits(3).build().unwrap();

    let outcome = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 3,
            minimum_resource: 6,
            maximum_resource: None,
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: false,
            random_seed: 0,
            error_score: SKErrorScore::Raise,
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();

    assert_eq!(outcome.rounds.len(), 1);
    assert_eq!(outcome.rounds[0].resource(), 6);
    assert_eq!(outcome.best_candidate_index, 0);
    assert_eq!(outcome.best_score, 0.0);
}
