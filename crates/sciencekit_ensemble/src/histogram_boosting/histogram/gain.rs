//! Numeric prefix scans and the Newton split-gain formula (spec
//! `histogram-gradient-boosting`, design decisions 3 and 5).

use bitvec::prelude::{BitVec, Lsb0};
use sciencekit_common::SKFloat;

use super::accumulate::SKFeatureHistogram;

/// The condition stored by an internal histogram-tree node.
#[derive(Debug, Clone, PartialEq)]
pub enum SKHistogramSplit {
    /// Numeric split: bins at or below `maximum_bin` go left; rows in the
    /// dedicated missing bin follow `default_left`.
    Numeric {
        /// The split feature.
        feature_index: usize,
        /// The largest bin routed left.
        maximum_bin: u8,
        /// Direction of rows in the missing bin.
        default_left: bool,
        /// The index of the dedicated missing bin.
        missing_bin: u8,
    },
    /// Categorical split: the categories whose membership bit is set go left;
    /// categories never observed at fit follow `unknown_left`.
    Categorical {
        /// The split feature.
        feature_index: usize,
        /// One membership bit per observed category bin.
        left_bins: BitVec<u8, Lsb0>,
        /// Direction of the unknown-category bin.
        unknown_left: bool,
    },
}

/// A split candidate together with its Newton gain.
#[derive(Debug, Clone, PartialEq)]
pub struct SKHistogramSplitCandidate<F> {
    /// The split condition.
    pub split: SKHistogramSplit,
    /// The gain `G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)`.
    pub gain: F,
}

/// Evaluate the Newton split gain for two disjoint children.
pub fn sk_split_gain<F: SKFloat>(
    left_gradient: F,
    left_hessian: F,
    right_gradient: F,
    right_hessian: F,
    total_gradient: F,
    total_hessian: F,
    l2_regularization: F,
) -> F {
    left_gradient * left_gradient / (left_hessian + l2_regularization)
        + right_gradient * right_gradient / (right_hessian + l2_regularization)
        - total_gradient * total_gradient / (total_hessian + l2_regularization)
}

/// Scan every numeric bin boundary of one feature, including both directions
/// for the missing bin, and return the largest valid gain.
///
/// A split is valid only when both children hold at least
/// `minimum_samples_leaf` rows and its gain is strictly positive.
pub fn sk_scan_numeric_feature<F: SKFloat>(
    feature_index: usize,
    histogram: &SKFeatureHistogram<F>,
    minimum_samples_leaf: usize,
    l2_regularization: F,
) -> Option<SKHistogramSplitCandidate<F>> {
    let stride = histogram.stride();
    let missing_bin = (stride - 1) as u8;
    let missing_gradient = histogram.gradient_sum(missing_bin);
    let missing_hessian = histogram.hessian_sum(missing_bin);
    let missing_count = histogram.count(missing_bin);
    let (total_gradient, total_hessian) = (
        histogram.total_gradient(),
        histogram.total_hessian(),
    );
    let mut best: Option<SKHistogramSplitCandidate<F>> = None;
    let mut left_gradient = F::zero();
    let mut left_hessian = F::zero();
    let mut left_count = 0usize;
    (0..stride - 1).for_each(|boundary| {
        let bin = boundary as u8;
        left_gradient = left_gradient + histogram.gradient_sum(bin);
        left_hessian = left_hessian + histogram.hessian_sum(bin);
        left_count += histogram.count(bin);
        [false, true].iter().for_each(|&default_left| {
            let (gradient_left, hessian_left, count_left) = if default_left {
                (
                    left_gradient + missing_gradient,
                    left_hessian + missing_hessian,
                    left_count + missing_count,
                )
            } else {
                (left_gradient, left_hessian, left_count)
            };
            let count_right = histogram.total_count() - count_left;
            if count_left < minimum_samples_leaf || count_right < minimum_samples_leaf {
                return;
            }
            let gain = sk_split_gain(
                gradient_left,
                hessian_left,
                total_gradient - gradient_left,
                total_hessian - hessian_left,
                total_gradient,
                total_hessian,
                l2_regularization,
            );
            if gain <= F::zero() {
                return;
            }
            let candidate = SKHistogramSplitCandidate {
                split: SKHistogramSplit::Numeric {
                    feature_index,
                    maximum_bin: bin,
                    default_left,
                    missing_bin,
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
    });
    best
}
