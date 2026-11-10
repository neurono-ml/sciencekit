//! Immutable ball-tree construction: the farthest-pivot split over the node
//! centroid.

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::super::node::SKBallTreeNode;
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
        let depth = super::node_builder::build_node(
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
