//! Caller-supplied leaf-value replacement preserving split geometry.

use sciencekit_common::{SKError, SKFloat};

use crate::node::SKTree;

use super::build::SKTreeStructure;

impl<F: SKFloat> SKTree<F> {
    /// The number of leaf nodes in the flat tree.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|node| node.is_leaf()).count()
    }

    /// Replace every leaf value in ascending leaf-node storage order while
    /// preserving all internal splits, thresholds, child links and counts.
    pub fn replace_leaf_values(&mut self, values: &[F]) -> Result<(), SKError> {
        let expected_leaves = self.leaf_count();
        if values.len() != expected_leaves {
            return Err(SKError::ShapeMismatch {
                expected: vec![expected_leaves],
                found: vec![values.len()],
            });
        }
        if let Some(position) = values.iter().position(|value| !value.is_finite()) {
            return Err(SKError::NonFiniteValue { position });
        }
        self.nodes
            .iter_mut()
            .filter(|node| node.is_leaf())
            .zip(values.iter())
            .for_each(|(leaf, &value)| {
                leaf.value = vec![value];
            });
        Ok(())
    }
}

impl<F: SKFloat> SKTreeStructure<F> {
    /// Replace every leaf value in ascending leaf-node storage order while
    /// preserving the split structure.
    pub fn replace_leaf_values(&mut self, values: &[F]) -> Result<(), SKError> {
        self.tree.replace_leaf_values(values)
    }
}
