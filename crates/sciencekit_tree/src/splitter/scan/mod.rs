//! Per-feature split scans for every supported criterion.
//!
//! This is a **pure dispatcher** module: `mod.rs` declares the submodules and
//! re-exports the scan entry point; the scans live in their own files.

mod absolute;
mod classification;
mod ranked;
mod squared;

use sciencekit_common::SKFloat;

use super::SKSplitCandidate;
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;

pub(crate) use classification::sk_scan_classification_feature;
pub(crate) use squared::sk_scan_squared_error_feature;

use absolute::sk_scan_absolute_error_feature;
use ranked::SKRankedSummary;

/// Whether the scan keeps the best threshold or draws one at random.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SKThresholdChoice {
    /// Evaluate every valid boundary and keep the best.
    Best,
    /// Draw one valid boundary from the seeded stream.
    Random(u64),
}

/// The criterion-specific inputs a scan needs.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SKScanContext<'a, F> {
    /// Classification statistics: canonical class index and weight per sample.
    Classification {
        /// The canonical class index of every training row.
        class_indices: &'a [usize],
        /// The sample weight of every training row.
        sample_weights: &'a [F],
        /// The number of fitted classes.
        number_of_classes: usize,
        /// Gini or entropy.
        criterion: SKTreeCriterion,
    },
    /// Regression statistics: the continuous response of every training row.
    Regression {
        /// The response of every training row.
        responses: &'a [F],
        /// Squared error or absolute error.
        criterion: SKTreeCriterion,
    },
}

/// Scan one feature and return its best (or single random) candidate.
pub(crate) fn sk_scan_feature<F: SKFloat>(
    design: &SKDesign<'_, F>,
    sample_indices: &[usize],
    feature: usize,
    minimum_samples_leaf: usize,
    context: SKScanContext<'_, F>,
    choice: SKThresholdChoice,
) -> Option<SKSplitCandidate<F>> {
    match context {
        SKScanContext::Classification {
            class_indices,
            sample_weights,
            number_of_classes,
            criterion,
        } => sk_scan_classification_feature(
            design,
            sample_indices,
            feature,
            minimum_samples_leaf,
            class_indices,
            sample_weights,
            number_of_classes,
            criterion,
            choice,
        ),
        SKScanContext::Regression {
            responses,
            criterion: SKTreeCriterion::SquaredError,
        } => sk_scan_squared_error_feature(
            design,
            sample_indices,
            feature,
            minimum_samples_leaf,
            responses,
            choice,
        ),
        SKScanContext::Regression {
            responses,
            criterion: SKTreeCriterion::AbsoluteError,
        } => sk_scan_absolute_error_feature(
            design,
            sample_indices,
            feature,
            minimum_samples_leaf,
            responses,
            choice,
        ),
        SKScanContext::Regression { .. } => None,
    }
}

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

/// Keep the strongest of the collected candidates.
pub(crate) fn sk_reduce_candidates<F: SKFloat>(
    candidates: impl IntoIterator<Item = SKSplitCandidate<F>>,
) -> Option<SKSplitCandidate<F>> {
    candidates
        .into_iter()
        .reduce(|current, challenger| sk_better_candidate(current, Some(challenger)))
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
