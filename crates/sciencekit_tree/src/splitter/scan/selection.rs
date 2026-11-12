//! Boundary selection helpers shared by the per-feature scans.

use sciencekit_common::SKFloat;

use super::SKThresholdChoice;
use super::ranked::SKRankedSummary;
use crate::splitter::SKSplitCandidate;

/// The midpoint between two distinct sorted feature values.
pub(crate) fn sk_midpoint<F: SKFloat>(left: F, right: F) -> F {
    left * F::from(0.5).expect("0.5 fits") + right * F::from(0.5).expect("0.5 fits")
}

/// Keep the stronger candidate, breaking ties by smaller feature and threshold.
pub(crate) fn sk_better_candidate<F: SKFloat>(
    current: SKSplitCandidate<F>,
    challenger: Option<SKSplitCandidate<F>>,
) -> SKSplitCandidate<F> {
    match challenger {
        Some(challenger) => {
            let challenger_wins = challenger.impurity_decrease() > current.impurity_decrease()
                || (challenger.impurity_decrease() == current.impurity_decrease()
                    && (challenger.feature() < current.feature()
                        || (challenger.feature() == current.feature()
                            && challenger.threshold() < current.threshold())));
            if challenger_wins { challenger } else { current }
        }
        None => current,
    }
}

/// Choose a valid boundary index, or the best one, from the scanned candidates.
pub(crate) fn sk_choose_boundary<F: SKFloat>(
    boundaries: &[(F, F)],
    choice: SKThresholdChoice,
) -> Option<(F, F)> {
    if boundaries.is_empty() {
        return None;
    }
    match choice {
        SKThresholdChoice::Best => boundaries.iter().copied().reduce(|current, challenger| {
            if challenger.1 > current.1 || (challenger.1 == current.1 && challenger.0 < current.0) {
                challenger
            } else {
                current
            }
        }),
        SKThresholdChoice::Random(seed) => {
            let mut generator = sciencekit_common::SKRandomGenerator::new(seed);
            Some(boundaries[generator.next_index(boundaries.len())])
        }
    }
}

/// A right-side summary rebuilt from a node's responses.
pub(crate) fn sk_summary_for_indices<F: SKFloat>(
    responses: &[F],
    sample_indices: &[usize],
) -> SKRankedSummary<F> {
    let mut values: Vec<F> = sample_indices.iter().map(|&row| responses[row]).collect();
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite responses"));
    values.dedup();
    let mut summary = SKRankedSummary::new(values);
    for &row in sample_indices {
        summary.insert(summary.rank_of(responses[row]));
    }
    summary
}
