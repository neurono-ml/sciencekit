//! `SKSplitter::Best`: feature-parallel exhaustive candidate evaluation.

use rayon::prelude::*;
use sciencekit_common::SKFloat;

use super::SKSplitCandidate;
use super::SKSplitter;
use super::scan::{SKScanContext, SKThresholdChoice, sk_better_candidate, sk_scan_feature};
use crate::design::SKDesign;

/// Evaluate every considered feature in parallel and keep the best candidate.
///
/// Ties break deterministically by smaller feature index, then smaller
/// threshold, so the result never depends on the thread count or on the order
/// in which features are supplied.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_best_split<F: SKFloat>(
    design: &SKDesign<'_, F>,
    sample_indices: &[usize],
    features: &[usize],
    minimum_samples_leaf: usize,
    context: SKScanContext<'_, F>,
    splitter: SKSplitter,
    node_seed: u64,
) -> Option<SKSplitCandidate<F>> {
    features
        .par_iter()
        .filter_map(|&feature| match splitter {
            SKSplitter::Best => sk_scan_feature(
                design,
                sample_indices,
                feature,
                minimum_samples_leaf,
                context,
                SKThresholdChoice::Best,
            ),
            SKSplitter::Random => {
                let seed = sciencekit_common::sk_derive_seed(
                    node_seed,
                    sciencekit_common::SK_RANDOM_DOMAIN_TREE,
                    feature as u64,
                );
                sk_scan_feature(
                    design,
                    sample_indices,
                    feature,
                    minimum_samples_leaf,
                    context,
                    SKThresholdChoice::Random(seed),
                )
            }
        })
        .reduce_with(|current, challenger| sk_better_candidate(current, Some(challenger)))
}
