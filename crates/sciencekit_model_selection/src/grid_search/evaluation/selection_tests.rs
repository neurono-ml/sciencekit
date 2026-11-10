//! Mean/standard-deviation aggregation and deterministic best selection
//! (spec `grid-search`, requirement `Score aggregation and best selection`).

use ndarray::{Array1, array};

use crate::grid_search::results::SKGridSearchResults;

fn results() -> SKGridSearchResults<f64> {
    SKGridSearchResults::from_scores(array![
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0],
        [0.0, 5.0, 1.0],
    ])
    .expect("the score matrix is non-empty")
}

#[test]
fn grid_selection_mean_and_population_deviation_match_hand_values() {
    let results = SKGridSearchResults::from_scores(array![
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0],
    ])
    .unwrap();
    assert_eq!(results.mean_scores(), Array1::from_vec(vec![2.0, 2.0]).view());
    let expected_deviation = (2.0_f64 / 3.0).sqrt();
    assert!((results.standard_deviations()[0] - expected_deviation).abs() < 1e-12);
    assert_eq!(results.standard_deviations()[1], 0.0);
}

#[test]
fn grid_selection_highest_mean_wins() {
    let results = SKGridSearchResults::from_scores(array![
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5],
    ])
    .unwrap();
    assert_eq!(results.best_index(), 1);
    assert_eq!(results.best_score(), 2.0);
    assert_eq!(results.ranked_candidate_indices(), vec![1, 0, 2]);
}

#[test]
fn grid_selection_ties_are_broken_by_candidate_index() {
    let results = results();
    assert_eq!(results.best_index(), 0);
    assert_eq!(results.ranked_candidate_indices(), vec![0, 1, 2]);
}

#[test]
fn grid_selection_rejects_an_empty_score_matrix() {
    let error = SKGridSearchResults::<f64>::from_scores(ndarray::Array2::zeros((0, 0)))
        .err()
        .expect("an empty matrix has no candidate");
    assert!(matches!(
        error,
        sciencekit_common::SKError::EmptyInput { .. }
    ));
}
