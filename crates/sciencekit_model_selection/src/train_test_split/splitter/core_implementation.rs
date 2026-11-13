//! The configured train/test splitter and its accessors.

use sciencekit_common::SKExecutionMode;

use super::super::builder::SKTrainTestSplitBuilder;
use super::super::configuration::SKTrainTestSplitConfiguration;
use super::super::partition_size::SKPartitionSize;

/// Splits features and optional targets into train and test partitions.
///
/// Constructed only through [`SKTrainTestSplitBuilder`]. With shuffling enabled
/// the order is a seeded permutation; the same seed always yields the same
/// partitions. When only one size is configured the other side receives the
/// remaining rows; when both are configured, the surplus rows are left unassigned.
/// With stratification enabled, each class contributes to both sides in
/// proportion to the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKTrainTestSplitter {
    pub(super) configuration: SKTrainTestSplitConfiguration,
}

impl SKTrainTestSplitter {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKTrainTestSplitBuilder {
        SKTrainTestSplitBuilder::new()
    }

    /// The configured test size, if any.
    pub fn test_size(&self) -> Option<SKPartitionSize> {
        self.configuration.test_size
    }

    /// The configured train size, if any.
    pub fn train_size(&self) -> Option<SKPartitionSize> {
        self.configuration.train_size
    }

    /// Whether a seeded shuffle precedes partitioning.
    pub fn shuffle(&self) -> bool {
        self.configuration.shuffle
    }

    /// The deterministic seed.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// Whether stratification is enabled.
    pub fn stratify(&self) -> bool {
        self.configuration.stratify
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }

    pub(crate) fn from_configuration(configuration: SKTrainTestSplitConfiguration) -> Self {
        Self { configuration }
    }
}
