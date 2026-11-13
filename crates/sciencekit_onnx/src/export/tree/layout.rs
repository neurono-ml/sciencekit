//! The fitted-tree layout read out of the neutral state.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::export::helpers::{floating_tensor, integer_tensor};

/// One fitted tree as the parallel node arrays the state carries.
pub(crate) struct SKOnnxTreeLayout<F: SKFloat> {
    pub(crate) split_feature: Vec<i64>,
    pub(crate) threshold: Vec<F>,
    pub(crate) left: Vec<i64>,
    pub(crate) right: Vec<i64>,
    pub(crate) values: Vec<F>,
    pub(crate) value_offsets: Vec<i64>,
}

impl<F: SKFloat> SKOnnxTreeLayout<F> {
    /// Read the tree under a tensor prefix (`tree` or a forest member prefix).
    pub(crate) fn read(state: &SKModelState<F>, prefix: &str) -> Result<Self, SKError> {
        let layout = SKOnnxTreeLayout {
            split_feature: integer_tensor(state, &format!("{prefix}.nodes.split_feature"))?,
            threshold: floating_tensor(state, &format!("{prefix}.nodes.threshold"))?.1,
            left: integer_tensor(state, &format!("{prefix}.nodes.left"))?,
            right: integer_tensor(state, &format!("{prefix}.nodes.right"))?,
            values: floating_tensor(state, &format!("{prefix}.nodes.value"))?.1,
            value_offsets: integer_tensor(state, &format!("{prefix}.nodes.value_offsets"))?,
        };
        layout.validate(state)?;
        Ok(layout)
    }

    /// Reject parallel arrays of different lengths and splits naming a feature
    /// the declared graph input does not have.
    fn validate(&self, state: &SKModelState<F>) -> Result<(), SKError> {
        let node_count = self.split_feature.len();
        for length in [self.threshold.len(), self.left.len(), self.right.len()] {
            if length != node_count {
                return Err(SKError::ShapeMismatch {
                    expected: vec![node_count],
                    found: vec![length],
                });
            }
        }
        if self.value_offsets.len() != node_count + 1 {
            return Err(SKError::ShapeMismatch {
                expected: vec![node_count + 1],
                found: vec![self.value_offsets.len()],
            });
        }
        let feature_count = state.metadata().feature_count() as i64;
        if let Some(&feature) = self
            .split_feature
            .iter()
            .find(|&&feature| feature >= feature_count)
        {
            return Err(SKError::ShapeMismatch {
                expected: vec![state.metadata().feature_count()],
                found: vec![feature.max(0) as usize],
            });
        }
        Ok(())
    }

    /// The leaf value slice of one node.
    pub(crate) fn leaf_values(&self, node: usize) -> &[F] {
        let start = self.value_offsets[node].max(0) as usize;
        let end = self.value_offsets[node + 1].max(0) as usize;
        &self.values[start..end.min(self.values.len())]
    }
}
