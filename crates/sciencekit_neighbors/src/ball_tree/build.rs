//! Immutable ball-tree construction: the farthest-pivot split over the node
//! centroid.

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::node::SKBallTreeNode;
use crate::metric::SKNeighborMetric;

/// An immutable ball-tree spatial index.
///
/// The tree owns a row-major copy of the training features, an index
/// permutation and a flat node array. It is `Send + Sync` and never mutates
/// after construction.
#[derive(Debug, Clone)]
pub struct SKBallTree<F> {
    pub(crate) training_features: Array2<F>,
    pub(crate) permutation: Vec<usize>,
    pub(crate) nodes: Vec<SKBallTreeNode<F>>,
    pub(crate) leaf_size: usize,
    pub(crate) metric: SKNeighborMetric,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_mode: SKExecutionMode,
    pub(crate) depth: usize,
}

impl<F: SKFloat> SKBallTree<F> {
    pub(crate) fn build_index(
        training_features: Array2<F>,
        leaf_size: usize,
        metric: SKNeighborMetric,
        backend: SKBackendKind,
        execution_mode: SKExecutionMode,
    ) -> SKBallTree<F> {
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
        SKBallTree {
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

    pub(crate) fn nodes(&self) -> &[SKBallTreeNode<F>] {
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
    nodes: &mut Vec<SKBallTreeNode<F>>,
) -> (usize, usize) {
    let centroid = centroid_of(features, permutation, start, end);
    let radius = permutation[start..end]
        .iter()
        .fold(F::zero(), |maximum, &row| {
            maximum.max(distance(features, row, &centroid))
        });
    let node_index = nodes.len();
    nodes.push(SKBallTreeNode::new(start, end, centroid, radius));

    if end - start <= leaf_size {
        return (node_index, 1);
    }

    let centroid = nodes[node_index].centroid().to_vec();
    let pivot_left = farthest_row(features, permutation, start, end, &centroid, None);
    let pivot_right = farthest_row(
        features,
        permutation,
        start,
        end,
        &features.row(pivot_left).to_vec(),
        Some(pivot_left),
    );
    let middle = start + (end - start) / 2;
    permutation[start..end].select_nth_unstable_by(middle - start, |&left, &right| {
        let left_gap = squared_distance(features, left, pivot_left);
        let right_gap = squared_distance(features, left, pivot_right);
        let left_other = squared_distance(features, right, pivot_left);
        let right_other = squared_distance(features, right, pivot_right);
        (left_gap - right_gap)
            .partial_cmp(&(left_other - right_other))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });

    let (left, left_depth) = build_node(features, permutation, start, middle, leaf_size, nodes);
    let (right, right_depth) = build_node(features, permutation, middle, end, leaf_size, nodes);
    nodes[node_index].left = Some(left);
    nodes[node_index].right = Some(right);
    (node_index, 1 + left_depth.max(right_depth))
}

fn centroid_of<F: SKFloat>(
    features: &Array2<F>,
    permutation: &[usize],
    start: usize,
    end: usize,
) -> Vec<F> {
    let features_count = features.ncols();
    let mut centroid = vec![F::zero(); features_count];
    for &row in &permutation[start..end] {
        for (column, &value) in features.row(row).iter().enumerate() {
            centroid[column] = centroid[column] + value;
        }
    }
    let count = F::from(end - start).expect("row count fits in the scalar");
    centroid.iter().map(|&total| total / count).collect()
}

fn farthest_row<F: SKFloat>(
    features: &Array2<F>,
    permutation: &[usize],
    start: usize,
    end: usize,
    target: &[F],
    excluded: Option<usize>,
) -> usize {
    let mut best_row = permutation[start];
    let mut best_distance = F::neg_infinity();
    for &row in &permutation[start..end] {
        if Some(row) == excluded {
            continue;
        }
        let candidate = distance(features, row, target);
        if candidate > best_distance {
            best_distance = candidate;
            best_row = row;
        }
    }
    best_row
}

fn distance<F: SKFloat>(features: &Array2<F>, row: usize, target: &[F]) -> F {
    features
        .row(row)
        .iter()
        .zip(target.iter())
        .fold(F::zero(), |total, (&value, &other)| {
            let difference = value - other;
            total + difference * difference
        })
        .sqrt()
}

fn squared_distance<F: SKFloat>(features: &Array2<F>, left: usize, right: usize) -> F {
    features
        .row(left)
        .iter()
        .zip(features.row(right).iter())
        .fold(F::zero(), |total, (&a, &b)| {
            let difference = a - b;
            total + difference * difference
        })
}
