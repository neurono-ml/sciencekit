//! TDD tests for deterministic candidate ranking, tie-breaking and round traces
//! (spec `halving-grid-search`, requirement `Deterministic elimination`).

use ndarray::{Array2, array};

use crate::grid_search::SKGridSearchResults;

use super::trace::{sk_build_round_trace, sk_rank_candidate_indices};

#[test]
fn halving_trace_ranks_by_descending_mean_and_breaks_ties_by_original_index() {
    let candidate_indices = vec![7, 2, 5, 1, 4];
    let mean_scores = array![0.80, 0.80, f64::NAN, 0.95, 0.80];

    let ranked = sk_rank_candidate_indices(&candidate_indices, mean_scores.view());
    assert_eq!(ranked, vec![1, 2, 4, 7, 5]);
}

#[test]
fn halving_trace_splits_survivors_and_eliminated_candidates_cleanly() {
    let scores = array![
        [0.40, 0.60],
        [0.80, 0.80],
        [0.90, 0.70],
        [0.10, 0.30],
        [0.95, 0.85],
        [0.50, 0.50],
    ];
    let results = SKGridSearchResults::from_scores(scores).unwrap();
    let round = sk_build_round_trace(0, 12, (0..6).collect(), results, 2, 2);

    assert_eq!(round.round(), 0);
    assert_eq!(round.resource(), 12);
    assert_eq!(round.evaluated_candidate_indices(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(round.surviving_candidate_indices(), &[4, 1]);
    assert_eq!(round.eliminated_candidate_indices(), &[2, 0, 5, 3]);
    assert!(round.aggressively_pruned_candidate_indices().is_empty());
    assert_eq!(round.ranked_candidate_indices(), vec![4, 1, 2, 0, 5, 3]);
    assert_eq!(round.fold_scores().dim(), (6, 2));
}

#[test]
fn halving_trace_tie_in_later_round_still_prefers_smaller_original_index() {
    // Suppose round 0 ranked candidate 8 ahead of candidate 3, so round 1 evaluates [8, 3].
    // When 8 and 3 tie in round 1, candidate 3 must win because its original index is smaller.
    let scores = Array2::from_elem((2, 3), 0.75_f64);
    let results = SKGridSearchResults::from_scores(scores).unwrap();
    let round = sk_build_round_trace(1, 36, vec![8, 3], results, 1, 1);

    assert_eq!(round.surviving_candidate_indices(), &[3]);
    assert_eq!(round.eliminated_candidate_indices(), &[8]);
}
