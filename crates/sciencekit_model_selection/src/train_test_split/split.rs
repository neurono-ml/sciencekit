//! The owned result of a train/test split.

use sciencekit_common::{SKFeatureBlock, SKFloat};

use crate::SKTargetPartition;

/// Features and optional targets gathered into train and test partitions.
///
/// The index vectors are preserved alongside the gathered data, so a caller can
/// inspect the partition or re-apply it to another aligned structure.
#[derive(Debug, Clone)]
pub struct SKTrainTestSplit<F: SKFloat> {
    train_features: SKFeatureBlock<F>,
    test_features: SKFeatureBlock<F>,
    train_targets: Option<SKTargetPartition<F>>,
    test_targets: Option<SKTargetPartition<F>>,
    train_indices: Vec<usize>,
    test_indices: Vec<usize>,
}

impl<F: SKFloat> SKTrainTestSplit<F> {
    pub(crate) fn new(
        train_features: SKFeatureBlock<F>,
        test_features: SKFeatureBlock<F>,
        train_targets: Option<SKTargetPartition<F>>,
        test_targets: Option<SKTargetPartition<F>>,
        train_indices: Vec<usize>,
        test_indices: Vec<usize>,
    ) -> Self {
        Self {
            train_features,
            test_features,
            train_targets,
            test_targets,
            train_indices,
            test_indices,
        }
    }

    /// The gathered train features.
    pub fn train_features(&self) -> &SKFeatureBlock<F> {
        &self.train_features
    }

    /// The gathered test features.
    pub fn test_features(&self) -> &SKFeatureBlock<F> {
        &self.test_features
    }

    /// The gathered train targets, when targets were supplied.
    pub fn train_targets(&self) -> Option<&SKTargetPartition<F>> {
        self.train_targets.as_ref()
    }

    /// The gathered test targets, when targets were supplied.
    pub fn test_targets(&self) -> Option<&SKTargetPartition<F>> {
        self.test_targets.as_ref()
    }

    /// The train row indices in the original dataset.
    pub fn train_indices(&self) -> &[usize] {
        &self.train_indices
    }

    /// The test row indices in the original dataset.
    pub fn test_indices(&self) -> &[usize] {
        &self.test_indices
    }

    /// The number of train rows.
    pub fn number_of_train_rows(&self) -> usize {
        self.train_indices.len()
    }

    /// The number of test rows.
    pub fn number_of_test_rows(&self) -> usize {
        self.test_indices.len()
    }
}
