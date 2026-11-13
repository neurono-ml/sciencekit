//! TDD tests for aggressive early elimination when resource rounds cannot support all survivors
//! (spec `halving-grid-search`, requirement `Aggressive elimination`).

use ndarray::{Array1, Array2};
use sciencekit_common::SKBuilder;

use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::grid_search::{SKErrorScore, SKRegressorScoring};
use crate::split::kfold::SKKFold;

use super::SKHalvingResource;
use super::evaluation::{SKHalvingEvaluationRequest, sk_run_halving_search};

#[test]
fn halving_aggressive_prunes_candidates_early_when_budget_cannot_support_survivors() {
    let features = Array2::zeros((30, 2));
    let targets = Array1::from_elem(30, 14.0);
    // 27 candidates with factor 3 normally need 3 rounds (27 -> 9 -> 3 -> 1),
    // but minimum_resource=10 and maximum_resource=30 only offer 2 rounds ([10, 30]).
    let candidates: Vec<SKMockConstantRegressor> = (0..27)
        .map(|index| SKMockConstantRegressor::new(index as f64))
        .collect();
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();

    let standard = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 3,
            minimum_resource: 10,
            maximum_resource: Some(30),
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: false,
            random_seed: 0,
            error_score: SKErrorScore::Raise,
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();

    let aggressive = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 3,
            minimum_resource: 10,
            maximum_resource: Some(30),
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: true,
            random_seed: 0,
            error_score: SKErrorScore::Raise,
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();

    assert_eq!(standard.rounds[0].surviving_candidate_indices().len(), 9);
    assert_eq!(standard.rounds[1].evaluated_candidate_indices().len(), 9);
    assert_eq!(standard.rounds[1].surviving_candidate_indices().len(), 3);

    // Aggressive mode prunes 27 -> 3 in round 0 (6 extra candidates pruned early)
    // so round 1 at resource 30 evaluates 3 candidates and finishes at 1 survivor.
    assert_eq!(aggressive.rounds.len(), 2);
    assert_eq!(aggressive.rounds[0].surviving_candidate_indices().len(), 3);
    assert_eq!(
        aggressive.rounds[0]
            .aggressively_pruned_candidate_indices()
            .len(),
        6
    );
    assert_eq!(aggressive.rounds[1].evaluated_candidate_indices().len(), 3);
    assert_eq!(aggressive.rounds[1].surviving_candidate_indices().len(), 1);
    assert_eq!(aggressive.best_candidate_index, 14);
}

#[test]
fn halving_aggressive_is_a_no_op_when_budget_supports_all_rounds_and_breaks_ties_deterministically()
{
    let features = Array2::zeros((90, 1));
    let targets = Array1::from_elem(90, 5.0);
    // All 9 candidates tie on round 0 except we put 4 tied winners at 5.0 (indices 1, 3, 5, 7)
    // with budget [10, 10] (only 1 round -> aggressive prunes 9 -> 1, picking index 1!).
    let candidates: Vec<SKMockConstantRegressor> = (0..9)
        .map(|index| {
            if index % 2 == 1 {
                SKMockConstantRegressor::new(5.0)
            } else {
                SKMockConstantRegressor::new(0.0)
            }
        })
        .collect();
    let splitter = SKKFold::build().number_of_splits(2).build().unwrap();

    let single_round_aggressive = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 3,
            minimum_resource: 10,
            maximum_resource: Some(10),
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: true,
            random_seed: 0,
            error_score: SKErrorScore::Raise,
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();
    assert_eq!(
        single_round_aggressive.rounds[0].surviving_candidate_indices(),
        &[1]
    );
    assert_eq!(
        single_round_aggressive.rounds[0].aggressively_pruned_candidate_indices(),
        &[3, 5]
    );

    // With full budget [10, 30, 90], 9 candidates fit cleanly without extra pruning.
    let full_budget = sk_run_halving_search(
        SKHalvingEvaluationRequest {
            candidates: &candidates,
            splitter: &splitter,
            features: features.view().try_into().unwrap(),
            targets: targets.view().try_into().unwrap(),
            groups: None,
            factor: 3,
            minimum_resource: 10,
            maximum_resource: Some(90),
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: true,
            random_seed: 0,
            error_score: SKErrorScore::Raise,
        },
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();
    assert!(
        full_budget.rounds[0]
            .aggressively_pruned_candidate_indices()
            .is_empty()
    );
    assert_eq!(full_budget.rounds[0].surviving_candidate_indices().len(), 3);
    assert_eq!(full_budget.best_candidate_index, 1);
}
