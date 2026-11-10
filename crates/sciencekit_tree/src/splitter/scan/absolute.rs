//! Absolute-error split scan over a sorted feature.

use sciencekit_common::SKFloat;

use super::ranked::SKRankedSummary;
use super::{SKThresholdChoice, sk_choose_boundary, sk_midpoint};
use crate::design::SKDesign;
use crate::splitter::SKSplitCandidate;

/// Scan one feature for the split minimizing the total absolute deviation,
/// or draw one random valid boundary.
pub(crate) fn sk_scan_absolute_error_feature<F: SKFloat>(
    design: &SKDesign<'_, F>,
    sample_indices: &[usize],
    feature: usize,
    minimum_samples_leaf: usize,
    responses: &[F],
    choice: SKThresholdChoice,
) -> Option<SKSplitCandidate<F>> {
    let sorted = design.sorted_feature_values(sample_indices, feature);
    let mut left = empty_summary(responses, sample_indices);
    let mut right = super::sk_summary_for_indices(responses, sample_indices);
    let parent_cost = right.absolute_deviation_cost();
    if parent_cost <= F::zero() {
        return None;
    }
    let mut boundaries = Vec::new();
    let mut position = 0;
    while position < sorted.len().saturating_sub(1) {
        let value = sorted[position].0;
        while position < sorted.len() && sorted[position].0 == value {
            let row = sorted[position].1;
            let rank = right.rank_of(responses[row]);
            right.remove(rank);
            left.insert(rank);
            position += 1;
        }
        let left_samples = position;
        let right_samples = sorted.len() - position;
        if left_samples < minimum_samples_leaf || right_samples < minimum_samples_leaf {
            continue;
        }
        let right_value = sorted[position].0;
        let left_cost = left.absolute_deviation_cost();
        let right_cost = right.absolute_deviation_cost();
        let decrease = parent_cost - left_cost - right_cost;
        if decrease > F::zero() {
            boundaries.push((sk_midpoint(value, right_value), decrease));
        }
    }
    let (threshold, decrease) = sk_choose_boundary(&boundaries, choice)?;
    Some(SKSplitCandidate::new(feature, threshold, decrease))
}

/// An empty summary over the node's response value table.
fn empty_summary<F: SKFloat>(responses: &[F], sample_indices: &[usize]) -> SKRankedSummary<F> {
    let mut values: Vec<F> = sample_indices.iter().map(|&row| responses[row]).collect();
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite responses"));
    values.dedup();
    SKRankedSummary::new(values)
}
