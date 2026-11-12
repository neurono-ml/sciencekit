//! Classification split scan over a sorted feature.

use sciencekit_common::SKFloat;

use super::{SKThresholdChoice, sk_midpoint};
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;
use crate::impurity::{sk_entropy_impurity, sk_gini_impurity};
use crate::splitter::SKSplitCandidate;

/// Scan one feature for the split maximizing the classification impurity
/// decrease, or draw one random valid boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_scan_classification_feature<F: SKFloat>(
    design: &SKDesign<'_, F>,
    sample_indices: &[usize],
    feature: usize,
    minimum_samples_leaf: usize,
    class_indices: &[usize],
    sample_weights: &[F],
    number_of_classes: usize,
    criterion: SKTreeCriterion,
    choice: SKThresholdChoice,
) -> Option<SKSplitCandidate<F>> {
    let sorted = design.sorted_feature_values(sample_indices, feature);
    let mut left_counts = vec![F::zero(); number_of_classes];
    let mut right_counts = vec![F::zero(); number_of_classes];
    for &row in sample_indices {
        right_counts[class_indices[row]] = right_counts[class_indices[row]] + sample_weights[row];
    }
    let parent_impurity = sk_classification_impurity(&right_counts, criterion);
    if parent_impurity <= F::zero() {
        return None;
    }
    let total_weight = right_counts
        .iter()
        .fold(F::zero(), |sum, value| sum + *value);
    let sample_count = sample_indices.len();
    let mut boundaries = Vec::new();
    for position in 0..sorted.len().saturating_sub(1) {
        let (value, row) = sorted[position];
        let class = class_indices[row];
        let weight = sample_weights[row];
        left_counts[class] = left_counts[class] + weight;
        right_counts[class] = right_counts[class] - weight;
        if value == sorted[position + 1].0 {
            continue;
        }
        let left_samples = position + 1;
        let right_samples = sample_count - left_samples;
        if left_samples < minimum_samples_leaf || right_samples < minimum_samples_leaf {
            continue;
        }
        let left_weight = left_counts
            .iter()
            .fold(F::zero(), |sum, value| sum + *value);
        let left_impurity = sk_classification_impurity(&left_counts, criterion);
        let right_impurity = sk_classification_impurity(&right_counts, criterion);
        let decrease = parent_impurity
            - (left_weight / total_weight) * left_impurity
            - ((total_weight - left_weight) / total_weight) * right_impurity;
        if decrease > F::zero() {
            boundaries.push((sk_midpoint(value, sorted[position + 1].0), decrease));
        }
    }
    let (threshold, decrease) = super::sk_choose_boundary(&boundaries, choice)?;
    Some(SKSplitCandidate::new(feature, threshold, decrease))
}

/// The classification impurity under the configured criterion.
pub(crate) fn sk_classification_impurity<F: SKFloat>(
    class_counts: &[F],
    criterion: SKTreeCriterion,
) -> F {
    match criterion {
        SKTreeCriterion::Entropy => sk_entropy_impurity(class_counts),
        _ => sk_gini_impurity(class_counts),
    }
}
