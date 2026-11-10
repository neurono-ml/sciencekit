//! The validated train/test split configuration.

use sciencekit_common::SKExecutionMode;

use super::partition_size::SKPartitionSize;

/// The immutable configuration carried by a built
/// [`super::SKTrainTestSplitter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKTrainTestSplitConfiguration {
    pub(crate) test_size: Option<SKPartitionSize>,
    pub(crate) train_size: Option<SKPartitionSize>,
    pub(crate) shuffle: bool,
    pub(crate) random_seed: u64,
    pub(crate) stratify: bool,
    pub(crate) execution_mode: SKExecutionMode,
}
