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
        Ok(SKOnnxTreeLayout {
            split_feature: integer_tensor(state, &format!("{prefix}.nodes.split_feature"))?,
            threshold: floating_tensor(state, &format!("{prefix}.nodes.threshold"))?.1,
            left: integer_tensor(state, &format!("{prefix}.nodes.left"))?,
            right: integer_tensor(state, &format!("{prefix}.nodes.right"))?,
            values: floating_tensor(state, &format!("{prefix}.nodes.value"))?.1,
            value_offsets: integer_tensor(state, &format!("{prefix}.nodes.value_offsets"))?,
        })
    }

    /// The leaf value slice of one node.
    pub(crate) fn leaf_values(&self, node: usize) -> &[F] {
        let start = self.value_offsets[node].max(0) as usize;
        let end = self.value_offsets[node + 1].max(0) as usize;
        &self.values[start..end.min(self.values.len())]
    }
}
