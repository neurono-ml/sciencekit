//! Squared-error split scan over a sorted feature.

use sciencekit_common::SKFloat;

use super::{SKThresholdChoice, sk_choose_boundary, sk_midpoint};
use crate::design::SKDesign;
use crate::splitter::SKSplitCandidate;

/// Scan one feature for the split maximizing the squared-error impurity
/// decrease, or draw one random valid boundary.
pub(crate) fn sk_scan_squared_error_feature<F: SKFloat>(
    design: &SKDesign<'_, F>,
    sample_indices: &[usize],
    feature: usize,
    minimum_samples_leaf: usize,
    responses: &[F],
    choice: SKThresholdChoice,
) -> Option<SKSplitCandidate<F>> {
    let sorted = design.sorted_feature_values(sample_indices, feature);
    let count = F::from(sample_indices.len()).expect("sample count fits in the scalar");
    let total_sum = sample_indices
        .iter()
        .fold(F::zero(), |sum, &row| sum + responses[row]);
    let total_squares = sample_indices
        .iter()
        .fold(F::zero(), |sum, &row| sum + responses[row] * responses[row]);
    let parent_impurity = sk_variance(total_sum, total_squares, count);
    if parent_impurity <= F::zero() {
        return None;
    }
    let mut left_sum = F::zero();
    let mut left_squares = F::zero();
    let mut boundaries = Vec::new();
    for position in 0..sorted.len().saturating_sub(1) {
        let (value, row) = sorted[position];
        left_sum = left_sum + responses[row];
        left_squares = left_squares + responses[row] * responses[row];
        if value == sorted[position + 1].0 {
            continue;
        }
        let left_samples = position + 1;
        let right_samples = sample_indices.len() - left_samples;
        if left_samples < minimum_samples_leaf || right_samples < minimum_samples_leaf {
            continue;
        }
        let left_count = F::from(left_samples).expect("count fits");
        let right_count = F::from(right_samples).expect("count fits");
        let left_impurity = sk_variance(left_sum, left_squares, left_count);
        let right_impurity = sk_variance(
            total_sum - left_sum,
            total_squares - left_squares,
            right_count,
        );
        let decrease = parent_impurity
            - (left_count / count) * left_impurity
            - (right_count / count) * right_impurity;
        if decrease > F::zero() {
            boundaries.push((sk_midpoint(value, sorted[position + 1].0), decrease));
        }
    }
    let (threshold, decrease) = sk_choose_boundary(&boundaries, choice)?;
    Some(SKSplitCandidate::new(feature, threshold, decrease))
}

/// Population variance from the sum and sum of squares.
fn sk_variance<F: SKFloat>(sum: F, sum_of_squares: F, count: F) -> F {
    let mean = sum / count;
    (sum_of_squares / count - mean * mean).max(F::zero())
}
