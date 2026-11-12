//! The splitter contract shared by every partitioning strategy.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat, SKTargetView};

use crate::index_split::SKIndexSplit;

/// Produces deterministic train/test partitions from a dataset.
///
/// A splitter receives optional targets and groups so one interface serves
/// unsupervised, stratified and grouped strategies. A strategy documents which
/// optional input it reads and validates the inputs it requires. Splitters
/// return indices, never data, so a consumer gathers only the rows a fold
/// actually needs.
pub trait SKSplitter<F: SKFloat> {
    /// Produces the list of index splits, in fold order.
    fn split(
        &self,
        features: SKDataView<'_, F>,
        targets: Option<SKTargetView<'_, F>>,
        groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError>;
}
