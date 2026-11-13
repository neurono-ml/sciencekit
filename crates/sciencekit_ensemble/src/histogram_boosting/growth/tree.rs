//! Flat histogram-tree storage and binned traversal (spec
//! `histogram-gradient-boosting`, design decision 8).

use sciencekit_common::SKFloat;

use crate::histogram_boosting::histogram::SKHistogramSplit;

/// One node of a fitted histogram tree.
#[derive(Debug, Clone, PartialEq)]
pub struct SKHistogramNode<F> {
    split: Option<SKHistogramSplit>,
    left_child: Option<usize>,
    right_child: Option<usize>,
    value: F,
    sample_count: usize,
}

impl<F: SKFloat> SKHistogramNode<F> {
    pub(super) fn internal(
        split: SKHistogramSplit,
        left_child: usize,
        right_child: usize,
        sample_count: usize,
    ) -> Self {
        Self {
            split: Some(split),
            left_child: Some(left_child),
            right_child: Some(right_child),
            value: F::zero(),
            sample_count,
        }
    }

    /// Build a leaf carrying its Newton value.
    pub fn leaf(sample_count: usize, value: F) -> Self {
        Self {
            split: None,
            left_child: None,
            right_child: None,
            value,
            sample_count,
        }
    }

    /// Whether this node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.split.is_none()
    }

    /// The split condition of an internal node.
    pub fn split(&self) -> Option<&SKHistogramSplit> {
        self.split.as_ref()
    }

    /// The left child index of an internal node.
    pub fn left_child(&self) -> Option<usize> {
        self.left_child
    }

    /// The right child index of an internal node.
    pub fn right_child(&self) -> Option<usize> {
        self.right_child
    }

    /// The learned Newton value of a leaf.
    pub fn value(&self) -> F {
        self.value
    }

    /// Number of training rows that reached the node.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

/// A fitted histogram tree stored as a flat node array.
#[derive(Debug, Clone, PartialEq)]
pub struct SKHistogramTree<F> {
    pub(super) nodes: Vec<SKHistogramNode<F>>,
    pub(super) feature_count: usize,
}

impl<F: SKFloat> SKHistogramTree<F> {
    pub(super) fn new(nodes: Vec<SKHistogramNode<F>>, feature_count: usize) -> Self {
        Self {
            nodes,
            feature_count,
        }
    }

    /// Every node in storage order (the root is node `0`).
    pub fn nodes(&self) -> &[SKHistogramNode<F>] {
        &self.nodes
    }

    /// A node by index.
    pub fn node(&self, index: usize) -> Option<&SKHistogramNode<F>> {
        self.nodes.get(index)
    }

    /// Number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of features the tree was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Traverse the tree with a per-feature bin lookup and return the leaf value.
    pub fn leaf_value_for(&self, bin_of_feature: impl Fn(usize) -> u8) -> F {
        let mut index = 0usize;
        loop {
            let node = &self.nodes[index];
            let Some(split) = node.split() else {
                return node.value();
            };
            index = match split {
                SKHistogramSplit::Numeric {
                    feature_index,
                    maximum_bin,
                    default_left,
                    missing_bin,
                } => {
                    let bin = bin_of_feature(*feature_index);
                    let goes_left = if bin == *missing_bin {
                        *default_left
                    } else {
                        bin <= *maximum_bin
                    };
                    if goes_left {
                        node.left_child().expect("internal node has a left child")
                    } else {
                        node.right_child().expect("internal node has a right child")
                    }
                }
                SKHistogramSplit::Categorical {
                    feature_index,
                    left_bins,
                    unknown_left,
                } => {
                    let bin = usize::from(bin_of_feature(*feature_index));
                    let goes_left = left_bins
                        .get(bin)
                        .map_or(*unknown_left, |membership| *membership);
                    if goes_left {
                        node.left_child().expect("internal node has a left child")
                    } else {
                        node.right_child().expect("internal node has a right child")
                    }
                }
            };
        }
    }
}
