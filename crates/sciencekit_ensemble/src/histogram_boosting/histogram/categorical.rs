//! Categorical category ordering and `bitvec` membership partitions (spec
//! `histogram-gradient-boosting`, design decision 6).

use bitvec::prelude::{BitVec, Lsb0};
use sciencekit_common::SKFloat;

use super::accumulate::SKFeatureHistogram;
use super::gain::{SKHistogramSplit, SKHistogramSplitCandidate, sk_split_gain};

/// Scan prefix partitions of the observed categories, ordered by their
/// gradient-to-Hessian ratio, and return the largest valid Newton gain.
///
/// The unknown-category bin is never part of a partition and always routes
/// right, which keeps prediction deterministic for categories unseen at fit.
pub fn sk_scan_categorical_feature<F: SKFloat>(
    feature_index: usize,
    histogram: &SKFeatureHistogram<F>,
    number_of_categories: usize,
    minimum_samples_leaf: usize,
    l2_regularization: F,
) -> Option<SKHistogramSplitCandidate<F>> {
    if number_of_categories < 2 {
        return None;
    }
    let mut ordering: Vec<usize> = (0..number_of_categories).collect();
    ordering.sort_by(|&left, &right| {
        let left_ratio = sk_gradient_ratio(histogram, left as u8, l2_regularization);
        let right_ratio = sk_gradient_ratio(histogram, right as u8, l2_regularization);
        right_ratio
            .partial_cmp(&left_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (total_gradient, total_hessian) = (histogram.total_gradient(), histogram.total_hessian());
    let mut best: Option<SKHistogramSplitCandidate<F>> = None;
    let mut left_gradient = F::zero();
    let mut left_hessian = F::zero();
    let mut left_count = 0usize;
    (0..number_of_categories - 1).for_each(|position| {
        let category = ordering[position];
        left_gradient = left_gradient + histogram.gradient_sum(category as u8);
        left_hessian = left_hessian + histogram.hessian_sum(category as u8);
        left_count += histogram.count(category as u8);
        let right_count = histogram.total_count() - left_count;
        if left_count < minimum_samples_leaf || right_count < minimum_samples_leaf {
            return;
        }
        let gain = sk_split_gain(
            left_gradient,
            left_hessian,
            total_gradient - left_gradient,
            total_hessian - left_hessian,
            total_gradient,
            total_hessian,
            l2_regularization,
        );
        if gain <= F::zero() {
            return;
        }
        let mut left_bins: BitVec<u8, Lsb0> = BitVec::repeat(false, number_of_categories);
        ordering[..=position].iter().for_each(|&left_category| {
            left_bins.set(left_category, true);
        });
        let candidate = SKHistogramSplitCandidate {
            split: SKHistogramSplit::Categorical {
                feature_index,
                left_bins,
                unknown_left: false,
            },
            gain,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.gain > current.gain)
        {
            best = Some(candidate);
        }
    });
    best
}

/// Gradient-to-Hessian ratio of one category bin, with a deterministic
/// infinity when the damped Hessian is zero.
fn sk_gradient_ratio<F: SKFloat>(
    histogram: &SKFeatureHistogram<F>,
    bin: u8,
    l2_regularization: F,
) -> F {
    let denominator = histogram.hessian_sum(bin) + l2_regularization;
    if denominator > F::zero() {
        histogram.gradient_sum(bin) / denominator
    } else if histogram.gradient_sum(bin) > F::zero() {
        F::infinity()
    } else if histogram.gradient_sum(bin) < F::zero() {
        F::neg_infinity()
    } else {
        F::zero()
    }
}
