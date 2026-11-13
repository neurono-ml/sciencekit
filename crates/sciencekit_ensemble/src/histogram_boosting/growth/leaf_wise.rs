//! Gain-ordered leaf-wise growth with histogram subtraction (spec
//! `histogram-gradient-boosting`, design decision 4).

use sciencekit_common::{SK_RANDOM_DOMAIN_TREE, SKFloat, SKRandomGenerator, sk_derive_seed};
use sciencekit_tree::SKMaximumFeatures;

use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::histogram::{
    SKFeatureHistogram, SKHistogramSplit, SKHistogramSplitCandidate, SKNodeHistogram,
    sk_accumulate_histogram, sk_scan_categorical_feature, sk_scan_numeric_feature,
    sk_subtract_histogram,
};

use super::tree::{SKHistogramNode, SKHistogramTree};

/// Growth controls of one histogram tree.
#[derive(Debug, Clone)]
pub struct SKHistogramGrowthParameters<F> {
    /// Total leaf budget (root included).
    pub maximum_leaf_nodes: usize,
    /// Lowest sample count allowed in either child.
    pub minimum_samples_leaf: usize,
    /// Hessian damping `λ` in `G / (H + λ)`.
    pub l2_regularization: F,
    /// Per-split feature-subsampling policy.
    pub maximum_features: SKMaximumFeatures<F>,
    /// Deterministic seed for feature subsampling.
    pub random_seed: u64,
}

/// A grown histogram tree and its normalized gain importances.
#[derive(Debug, Clone)]
pub struct SKHistogramGrowthResult<F> {
    /// The fitted tree.
    pub tree: SKHistogramTree<F>,
    /// Normalized per-feature gain importances.
    pub feature_importances: Vec<F>,
}

struct SKPendingLeaf<F> {
    node_index: usize,
    rows: Vec<usize>,
    histogram: SKNodeHistogram<F>,
    candidate: Option<SKHistogramSplitCandidate<F>>,
}

impl<F: SKFloat> SKPendingLeaf<F> {
    fn gain(&self) -> F {
        self.candidate
            .as_ref()
            .map_or(F::zero(), |candidate| candidate.gain)
    }
}

/// Grow one histogram tree leaf-wise, always splitting the leaf whose own best
/// split has the largest gain until the leaf budget is exhausted.
pub fn sk_grow_histogram_tree<F: SKFloat>(
    dataset: &SKBinnedDataset<F>,
    negative_gradients: &[F],
    hessians: &[F],
    rows: &[usize],
    parameters: &SKHistogramGrowthParameters<F>,
) -> SKHistogramGrowthResult<F> {
    let mut nodes: Vec<SKHistogramNode<F>> = Vec::with_capacity(parameters.maximum_leaf_nodes * 2);
    let mut gain_totals = vec![F::zero(); dataset.number_of_features()];
    let root_histogram = sk_accumulate_histogram(dataset, rows, negative_gradients, hessians);
    nodes.push(SKHistogramNode::leaf(rows.len(), F::zero()));
    let mut pending = vec![SKPendingLeaf {
        node_index: 0,
        rows: rows.to_vec(),
        candidate: sk_best_candidate(dataset, &root_histogram, parameters, 0),
        histogram: root_histogram,
    }];
    let mut leaf_count = 1usize;
    while leaf_count < parameters.maximum_leaf_nodes && !pending.is_empty() {
        let best_position = pending
            .iter()
            .enumerate()
            .filter(|(_, leaf)| leaf.candidate.is_some())
            .max_by(|(_, left), (_, right)| {
                left.gain()
                    .partial_cmp(&right.gain())
                    .expect("finite gains")
                    .then_with(|| right.node_index.cmp(&left.node_index))
            })
            .map(|(position, _)| position);
        let Some(position) = best_position else {
            break;
        };
        let leaf = pending.swap_remove(position);
        let candidate = leaf.candidate.expect("selected leaves are splittable");
        let split = candidate.split.clone();
        let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = leaf
            .rows
            .iter()
            .partition(|&&row| sk_row_goes_left(dataset, &split, row));
        let (right_histogram, left_histogram) = if right_rows.len() <= left_rows.len() {
            let right = sk_accumulate_histogram(dataset, &right_rows, negative_gradients, hessians);
            let left = sk_subtract_histogram(&leaf.histogram, &right);
            (right, left)
        } else {
            let left = sk_accumulate_histogram(dataset, &left_rows, negative_gradients, hessians);
            let right = sk_subtract_histogram(&leaf.histogram, &left);
            (right, left)
        };
        let left_index = nodes.len();
        let right_index = left_index + 1;
        let left_count = left_rows.len();
        let right_count = right_rows.len();
        nodes.push(SKHistogramNode::leaf(
            left_count,
            sk_leaf_value(left_histogram.feature(0), parameters.l2_regularization),
        ));
        nodes.push(SKHistogramNode::leaf(
            right_count,
            sk_leaf_value(right_histogram.feature(0), parameters.l2_regularization),
        ));
        nodes[leaf.node_index] =
            SKHistogramNode::internal(split, left_index, right_index, leaf.rows.len());
        gain_totals[candidate.feature_index()] =
            gain_totals[candidate.feature_index()] + candidate.gain;
        leaf_count += 1;
        pending.push(SKPendingLeaf {
            node_index: left_index,
            rows: left_rows,
            candidate: sk_best_candidate(dataset, &left_histogram, parameters, left_index),
            histogram: left_histogram,
        });
        pending.push(SKPendingLeaf {
            node_index: right_index,
            rows: right_rows,
            candidate: sk_best_candidate(dataset, &right_histogram, parameters, right_index),
            histogram: right_histogram,
        });
    }
    pending
        .iter()
        .for_each(|leaf| finalize_leaf(&mut nodes, leaf, parameters.l2_regularization));
    let normalizer = gain_totals
        .iter()
        .fold(F::zero(), |sum, &value| sum + value);
    let feature_importances = if normalizer > F::zero() {
        gain_totals
            .iter()
            .map(|&value| value / normalizer)
            .collect()
    } else {
        vec![F::zero(); dataset.number_of_features()]
    };
    SKHistogramGrowthResult {
        tree: SKHistogramTree::new(nodes, dataset.number_of_features()),
        feature_importances,
    }
}

fn finalize_leaf<F: SKFloat>(
    nodes: &mut [SKHistogramNode<F>],
    leaf: &SKPendingLeaf<F>,
    l2_regularization: F,
) {
    let value = sk_leaf_value(leaf.histogram.feature(0), l2_regularization);
    if let Some(node) = nodes.get_mut(leaf.node_index) {
        if node.is_leaf() {
            *node = SKHistogramNode::leaf(leaf.rows.len(), value);
        }
    }
}

fn sk_leaf_value<F: SKFloat>(histogram: &SKFeatureHistogram<F>, l2_regularization: F) -> F {
    let denominator = histogram.total_hessian() + l2_regularization;
    if denominator <= F::zero() {
        F::zero()
    } else {
        histogram.total_gradient() / denominator
    }
}

fn sk_best_candidate<F: SKFloat>(
    dataset: &SKBinnedDataset<F>,
    histogram: &SKNodeHistogram<F>,
    parameters: &SKHistogramGrowthParameters<F>,
    node_index: usize,
) -> Option<SKHistogramSplitCandidate<F>> {
    let features = sk_sampled_features(parameters, histogram.number_of_features(), node_index);
    features
        .into_iter()
        .filter_map(|feature| {
            let binning = dataset.feature_binning(feature);
            if binning.is_categorical() {
                sk_scan_categorical_feature(
                    feature,
                    histogram.feature(feature),
                    binning.number_of_bins(),
                    parameters.minimum_samples_leaf,
                    parameters.l2_regularization,
                )
            } else {
                sk_scan_numeric_feature(
                    feature,
                    histogram.feature(feature),
                    parameters.minimum_samples_leaf,
                    parameters.l2_regularization,
                )
            }
        })
        .max_by(|left, right| {
            left.gain
                .partial_cmp(&right.gain)
                .expect("finite gains")
                .then_with(|| left.feature_index().cmp(&right.feature_index()))
        })
}

fn sk_sampled_features<F: SKFloat>(
    parameters: &SKHistogramGrowthParameters<F>,
    feature_count: usize,
    node_index: usize,
) -> Vec<usize> {
    let resolved = parameters.maximum_features.resolved_count(feature_count);
    if resolved >= feature_count {
        return (0..feature_count).collect();
    }
    let seed = sk_derive_seed(
        parameters.random_seed,
        SK_RANDOM_DOMAIN_TREE,
        node_index as u64,
    );
    let mut features: Vec<usize> = SKRandomGenerator::new(seed)
        .shuffled_indices(feature_count)
        .into_iter()
        .take(resolved)
        .collect();
    features.sort_unstable();
    features
}

fn sk_row_goes_left<F: SKFloat>(
    dataset: &SKBinnedDataset<F>,
    split: &SKHistogramSplit,
    row: usize,
) -> bool {
    match split {
        SKHistogramSplit::Numeric {
            feature_index,
            maximum_bin,
            default_left,
            missing_bin,
        } => {
            let bin = dataset.bin(*feature_index, row);
            if bin == *missing_bin {
                *default_left
            } else {
                bin <= *maximum_bin
            }
        }
        SKHistogramSplit::Categorical {
            feature_index,
            left_bins,
            unknown_left,
        } => {
            let bin = usize::from(dataset.bin(*feature_index, row));
            left_bins
                .get(bin)
                .map_or(*unknown_left, |membership| *membership)
        }
    }
}

impl<F> SKHistogramSplitCandidate<F> {
    fn feature_index(&self) -> usize {
        match &self.split {
            SKHistogramSplit::Numeric { feature_index, .. }
            | SKHistogramSplit::Categorical { feature_index, .. } => *feature_index,
        }
    }
}
