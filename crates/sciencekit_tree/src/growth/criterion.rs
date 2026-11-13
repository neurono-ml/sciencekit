//! The criterion contract the shared growth engine drives.

use sciencekit_common::SKFloat;

use crate::splitter::SKScanContext;

/// What the growth engine needs from a classification or regression criterion.
pub(crate) trait SKTreeCriterion<F: SKFloat>: Sync {
    /// The impurity of a node holding the given training rows.
    fn impurity(&self, sample_indices: &[usize]) -> F;

    /// The learned output of a leaf holding the given training rows.
    fn leaf_output(&self, sample_indices: &[usize]) -> Vec<F>;

    /// The statistics the split scan accumulates.
    fn scan_context(&self) -> SKScanContext<'_, F>;
}
