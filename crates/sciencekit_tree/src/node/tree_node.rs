//! The node of a fitted tree.

use sciencekit_common::SKFloat;

/// One node of a fitted tree.
///
/// Nodes live in a flat contiguous vector owned by [`super::SKTree`]; children
/// are referenced by index. An internal node stores the split feature and
/// threshold; a leaf stores the learned output.
#[derive(Debug, Clone, PartialEq)]
pub struct SKTreeNode<F> {
    pub(crate) split_feature: Option<usize>,
    pub(crate) threshold: Option<F>,
    pub(crate) left_child: Option<usize>,
    pub(crate) right_child: Option<usize>,
    pub(crate) sample_count: usize,
    pub(crate) impurity: F,
    pub(crate) value: Vec<F>,
}

impl<F: SKFloat> SKTreeNode<F> {
    /// Build an internal split node.
    pub(crate) fn split(
        split_feature: usize,
        threshold: F,
        left_child: usize,
        right_child: usize,
        sample_count: usize,
        impurity: F,
    ) -> Self {
        SKTreeNode {
            split_feature: Some(split_feature),
            threshold: Some(threshold),
            left_child: Some(left_child),
            right_child: Some(right_child),
            sample_count,
            impurity,
            value: Vec::new(),
        }
    }

    /// Build a leaf node carrying its learned output.
    pub(crate) fn leaf(sample_count: usize, impurity: F, value: Vec<F>) -> Self {
        SKTreeNode {
            split_feature: None,
            threshold: None,
            left_child: None,
            right_child: None,
            sample_count,
            impurity,
            value,
        }
    }

    /// Whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.split_feature.is_none()
    }

    /// The feature this node splits on, or `None` for a leaf.
    pub fn split_feature(&self) -> Option<usize> {
        self.split_feature
    }

    /// The threshold this node splits on, or `None` for a leaf.
    pub fn threshold(&self) -> Option<F> {
        self.threshold
    }

    /// The index of the left child, or `None` for a leaf.
    pub fn left_child(&self) -> Option<usize> {
        self.left_child
    }

    /// The index of the right child, or `None` for a leaf.
    pub fn right_child(&self) -> Option<usize> {
        self.right_child
    }

    /// The number of training samples that reached the node.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// The impurity of the node under the configured criterion.
    pub fn impurity(&self) -> F {
        self.impurity
    }

    /// The learned output of a leaf.
    ///
    /// A classifier leaf stores its normalized class distribution (one entry
    /// per fitted class); a regressor leaf stores a single response.
    pub fn value(&self) -> &[F] {
        &self.value
    }
}
