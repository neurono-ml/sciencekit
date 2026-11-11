//! Round trace and deterministic candidate elimination
//! (spec `halving-grid-search`, requirement `Deterministic elimination`).

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;

use crate::grid_search::{SKGridSearchResults, sk_compare_scores};

/// The immutable record of one successive-halving evaluation round.
#[derive(Debug, Clone, PartialEq)]
pub struct SKHalvingRound<F: SKFloat> {
    round: usize,
    resource: usize,
    evaluated_candidate_indices: Vec<usize>,
    surviving_candidate_indices: Vec<usize>,
    eliminated_candidate_indices: Vec<usize>,
    aggressively_pruned_candidate_indices: Vec<usize>,
    results: SKGridSearchResults<F>,
}

impl<F: SKFloat> SKHalvingRound<F> {
    /// The zero-based round number.
    pub fn round(&self) -> usize {
        self.round
    }

    /// The resource budget evaluated in this round.
    pub fn resource(&self) -> usize {
        self.resource
    }

    /// Original candidate indices evaluated in this round.
    pub fn evaluated_candidate_indices(&self) -> &[usize] {
        &self.evaluated_candidate_indices
    }

    /// Original candidate indices surviving into the next round, ranked best-first.
    pub fn surviving_candidate_indices(&self) -> &[usize] {
        &self.surviving_candidate_indices
    }

    /// Original candidate indices eliminated at the end of this round.
    pub fn eliminated_candidate_indices(&self) -> &[usize] {
        &self.eliminated_candidate_indices
    }

    /// Candidates pruned beyond `ceil(n / factor)` when aggressive elimination is active.
    pub fn aggressively_pruned_candidate_indices(&self) -> &[usize] {
        &self.aggressively_pruned_candidate_indices
    }

    /// All evaluated original candidate indices ranked by descending mean score,
    /// with ties broken by ascending original candidate index.
    pub fn ranked_candidate_indices(&self) -> Vec<usize> {
        sk_rank_candidate_indices(
            &self.evaluated_candidate_indices,
            self.results.mean_scores(),
        )
    }

    /// Mean cross-validated score for each candidate in [`Self::evaluated_candidate_indices`].
    pub fn mean_scores(&self) -> ArrayView1<'_, F> {
        self.results.mean_scores()
    }

    /// Population standard deviation for each candidate in [`Self::evaluated_candidate_indices`].
    pub fn standard_deviations(&self) -> ArrayView1<'_, F> {
        self.results.standard_deviations()
    }

    /// The `(evaluated_candidates × folds)` score matrix for this round.
    pub fn fold_scores(&self) -> ArrayView2<'_, F> {
        self.results.fold_scores()
    }

    /// The underlying cross-validation results of this round.
    pub fn results(&self) -> &SKGridSearchResults<F> {
        &self.results
    }
}

/// Ranks `candidate_indices` by descending `mean_scores` with ties broken by
/// ascending original candidate index, placing `NaN` scores below finite scores.
pub(crate) fn sk_rank_candidate_indices<F: SKFloat>(
    candidate_indices: &[usize],
    mean_scores: ArrayView1<'_, F>,
) -> Vec<usize> {
    let mut positions: Vec<usize> = (0..candidate_indices.len()).collect();
    positions.sort_by(|&left, &right| {
        sk_compare_scores(mean_scores[right], mean_scores[left])
            .then_with(|| candidate_indices[left].cmp(&candidate_indices[right]))
    });
    positions
        .into_iter()
        .map(|position| candidate_indices[position])
        .collect()
}

/// Builds the round trace entry from the evaluated candidates and their fold results.
#[allow(dead_code)]
pub(crate) fn sk_build_round_trace<F: SKFloat>(
    round: usize,
    resource: usize,
    evaluated_candidate_indices: Vec<usize>,
    results: SKGridSearchResults<F>,
    standard_survivor_count: usize,
    surviving_count: usize,
) -> SKHalvingRound<F> {
    let ranked = sk_rank_candidate_indices(&evaluated_candidate_indices, results.mean_scores());
    let bounded_surviving = surviving_count.min(ranked.len());
    let bounded_standard = standard_survivor_count
        .min(ranked.len())
        .max(bounded_surviving);
    let surviving_candidate_indices = ranked[..bounded_surviving].to_vec();
    let eliminated_candidate_indices = ranked[bounded_surviving..].to_vec();
    let aggressively_pruned_candidate_indices =
        ranked[bounded_surviving..bounded_standard].to_vec();

    SKHalvingRound {
        round,
        resource,
        evaluated_candidate_indices,
        surviving_candidate_indices,
        eliminated_candidate_indices,
        aggressively_pruned_candidate_indices,
        results,
    }
}
