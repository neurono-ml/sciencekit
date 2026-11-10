//! Immutable KD-tree construction: the array-based node store and the median
//! split over the widest-spread feature.

use std::cmp::Ordering;

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::node::SKKDTreeNode;
use crate::metric::SKNeighborMetric;

/// An immutable KD-tree spatial index.
///
/// The tree owns a row-major copy of the training features, an index
/// permutation and a flat node array. It is `Send + Sync` and never mutates
/// after construction.
#[derive(Debug, Clone)]
pub struct SKKDTree<F> {
    pub(crate) training_features: Array2<F>,
    pub(crate) permutation: Vec<usize>,
    pub(crate) nodes: Vec<SKKDTreeNode<F>>,
    pub(crate) leaf_size: usize,
    pub(crate) metric: SKNeighborMetric,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_mode: SKExecutionMode,
    pub(crate) depth: usize,
}

impl<F: SKFloat> SKKDTree<F> {
    pub(crate) fn build_index(
        training_features: Array2<F>,
        leaf_size: usize,
        metric: SKNeighborMetric,
        backend: SKBackendKind,
        execution_mode: SKExecutionMode,
    ) -> SKKDTree<F> {
        let rows = training_features.nrows();
        let mut permutation: Vec<usize> = (0..rows).collect();
        let mut nodes = Vec::new();
        let depth = build_node(
            &training_features,
            &mut permutation,
            0,
            rows,
            leaf_size,
            &mut nodes,
        )
        .1;
        SKKDTree {
            training_features,
            permutation,
            nodes,
            leaf_size,
            metric,
            backend,
            execution_mode,
            depth,
        }
    }

    /// The configured maximum number of rows in a leaf.
    pub fn leaf_size(&self) -> usize {
        self.leaf_size
    }

    /// The configured distance metric.
    pub fn metric(&self) -> SKNeighborMetric {
        self.metric
    }

    /// The backend kind recorded on build and query spans and carried into
    /// downstream model configuration.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }

    /// The number of training rows the index was built from.
    pub fn training_rows(&self) -> usize {
        self.training_features.nrows()
    }

    /// The number of features per training row.
    pub fn training_features(&self) -> usize {
        self.training_features.ncols()
    }

    /// The number of nodes in the flat node store.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The maximum root-to-leaf node count.
    pub fn depth(&self) -> usize {
        self.depth
    }

    pub(crate) fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    pub(crate) fn nodes(&self) -> &[SKKDTreeNode<F>] {
        &self.nodes
    }
}

/// Build the subtree over `[start, end)`, returning its node index and depth.
fn build_node<F: SKFloat>(
    features: &Array2<F>,
    permutation: &mut [usize],
    start: usize,
    end: usize,
    leaf_size: usize,
    nodes: &mut Vec<SKKDTreeNode<F>>,
) -> (usize, usize) {
    let (minimum, maximum) = node_bounds(features, permutation, start, end);
    let node_index = nodes.len();
    nodes.push(SKKDTreeNode::new(start, end, minimum, maximum));

    if end - start <= leaf_size {
        return (node_index, 1);
    }

    let split_feature = widest_spread_feature(nodes[node_index].minimum(), nodes[node_index].maximum());
    let middle = start + (end - start) / 2;
    permutation[start..end].select_nth_unstable_by(middle - start, |&left, &right| {
        compare_rows(features, left, right, split_feature)
    });

    let (left, left_depth) = build_node(features, permutation, start, middle, leaf_size, nodes);
    let (right, right_depth) = build_node(features, permutation, middle, end, leaf_size, nodes);
    nodes[node_index].left = Some(left);
    nodes[node_index].right = Some(right);
    (node_index, 1 + left_depth.max(right_depth))
}

fn node_bounds<F: SKFloat>(
    features: &Array2<F>,
    permutation: &[usize],
    start: usize,
    end: usize,
) -> (Vec<F>, Vec<F>) {
    let features_count = features.ncols();
    let mut minimum = vec![F::infinity(); features_count];
    let mut maximum = vec![F::neg_infinity(); features_count];
    for &row in &permutation[start..end] {
        for (column, &value) in features.row(row).iter().enumerate() {
            minimum[column] = minimum[column].min(value);
            maximum[column] = maximum[column].max(value);
        }
    }
    (minimum, maximum)
}

fn widest_spread_feature<F: SKFloat>(minimum: &[F], maximum: &[F]) -> usize {
    let mut best_feature = 0;
    let mut best_spread = maximum[0] - minimum[0];
    for feature in 1..minimum.len() {
        let spread = maximum[feature] - minimum[feature];
        if spread > best_spread {
            best_spread = spread;
            best_feature = feature;
        }
    }
    best_feature
}

fn compare_rows<F: SKFloat>(
    features: &Array2<F>,
    left: usize,
    right: usize,
    feature: usize,
) -> Ordering {
    features[[left, feature]]
        .partial_cmp(&features[[right, feature]])
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.cmp(&right))
}
