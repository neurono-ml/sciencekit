//! The K-fold splitter.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKExecutionMode, SKFloat, SKTargetView};

use super::builder::SKKFoldBuilder;
use super::configuration::SKKFoldConfiguration;
use crate::{SKIndexSplit, SKSplitter, sk_shuffle_indices};

use crate::split::validation::{
    sk_number_of_rows, sk_partition_into_chunks, sk_split_from_test,
    sk_validate_finite_features, sk_validate_finite_targets, sk_validate_non_empty,
    sk_validate_number_of_splits,
};

/// Plain K-fold, constructed only through [`SKKFoldBuilder`].
///
/// Every fold's test partition is disjoint from the others and the test
/// partitions together cover every row exactly once. Optional targets and groups
/// are accepted by the contract and ignored, so plain K-fold behaves identically
/// whichever optional inputs a caller supplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKKFold {
    configuration: SKKFoldConfiguration,
}

impl SKKFold {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKKFoldBuilder {
        SKKFoldBuilder::new()
    }

    /// The configured number of folds.
    pub fn number_of_splits(&self) -> usize {
        self.configuration.number_of_splits
    }

    /// Whether a seeded shuffle precedes partitioning.
    pub fn shuffle(&self) -> bool {
        self.configuration.shuffle
    }

    /// The deterministic seed.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }

    pub(super) fn from_configuration(configuration: SKKFoldConfiguration) -> Self {
        Self { configuration }
    }

    fn ordered_rows(&self, number_of_rows: usize) -> Vec<usize> {
        if self.configuration.shuffle {
            sk_shuffle_indices(self.configuration.random_seed, number_of_rows)
        } else {
            (0..number_of_rows).collect()
        }
    }
}

impl<F: SKFloat> SKSplitter<F> for SKKFold {
    fn split(
        &self,
        features: SKDataView<'_, F>,
        _targets: Option<SKTargetView<'_, F>>,
        _groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError> {
        let number_of_rows = sk_number_of_rows(&features);
        sk_validate_non_empty(number_of_rows, "SKKFold::split")?;
        sk_validate_finite_features(&features)?;
        sk_validate_finite_targets(_targets)?;
        sk_validate_number_of_splits(self.configuration.number_of_splits, number_of_rows)?;

        let order = self.ordered_rows(number_of_rows);
        let chunks = sk_partition_into_chunks(&order, self.configuration.number_of_splits);
        Ok(chunks
            .into_iter()
            .map(|test| sk_split_from_test(number_of_rows, test))
            .collect())
    }
}
