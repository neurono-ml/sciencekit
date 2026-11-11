//! The flat tree owner and its traversal helper.

use sciencekit_common::SKFloat;

use super::SKTreeNode;

/// A fitted binary tree stored as a flat node array plus a root index.
#[derive(Debug, Clone, PartialEq)]
pub struct SKTree<F> {
    pub(crate) nodes: Vec<SKTreeNode<F>>,
    pub(crate) root: usize,
    pub(crate) feature_count: usize,
}

impl<F: SKFloat> SKTree<F> {
    /// Assemble a finished tree.
    pub fn new(nodes: Vec<SKTreeNode<F>>, feature_count: usize) -> Self {
        SKTree {
            nodes,
            root: 0,
            feature_count,
        }
    }

    /// Every node, in storage order.
    pub fn nodes(&self) -> &[SKTreeNode<F>] {
        &self.nodes
    }

    /// The index of the root node.
    pub fn root_index(&self) -> usize {
        self.root
    }

    /// A node by index.
    pub fn node(&self, index: usize) -> Option<&SKTreeNode<F>> {
        self.nodes.get(index)
    }

    /// The number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The number of features the tree was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// The maximum number of edges from the root to a leaf.
    pub fn depth(&self) -> usize {
        let mut maximum = 0;
        let mut stack = vec![(self.root, 0usize)];
        while let Some((index, depth)) = stack.pop() {
            maximum = maximum.max(depth);
            let node = &self.nodes[index];
            if let (Some(left), Some(right)) = (node.left_child(), node.right_child()) {
                stack.push((left, depth + 1));
                stack.push((right, depth + 1));
            }
        }
        maximum
    }

    /// The leaf a sample reaches, reading feature values through `feature_value`.
    ///
    /// A value less than or equal to the node threshold goes left; every value
    /// is expected to be finite (the public operations validate that first).
    pub fn leaf_index_for(&self, feature_value: impl Fn(usize) -> F) -> usize {
        self.leaf_and_depth_for(feature_value).0
    }

    /// The `(leaf_index, edge_depth)` pair a sample reaches when traversing from the root.
    pub fn leaf_and_depth_for(&self, feature_value: impl Fn(usize) -> F) -> (usize, usize) {
        let mut index = self.root;
        let mut depth = 0usize;
        loop {
            let node = &self.nodes[index];
            match (
                node.split_feature(),
                node.threshold(),
                node.left_child(),
                node.right_child(),
            ) {
                (Some(feature), Some(threshold), Some(left), Some(right)) => {
                    index = if feature_value(feature) <= threshold {
                        left
                    } else {
                        right
                    };
                    depth += 1;
                }
                _ => return (index, depth),
            }
        }
    }
}
