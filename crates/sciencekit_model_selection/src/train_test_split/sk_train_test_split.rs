//! The public train/test split entry point.

use sciencekit_common::{SKDataView, SKError, SKFloat, SKTargetView};

use super::split::SKTrainTestSplit;
use super::splitter::SKTrainTestSplitter;
use crate::{SKSplitter, sk_gather_rows};

/// Splits features and optional targets into train and test partitions.
///
/// The splitter supplies the indices; this helper gathers the selected rows into
/// owned blocks without ever densifying a sparse input.
pub fn sk_train_test_split<F: SKFloat>(
    splitter: &SKTrainTestSplitter,
    features: SKDataView<'_, F>,
    targets: Option<SKTargetView<'_, F>>,
) -> Result<SKTrainTestSplit<F>, SKError> {
    let mut splits = splitter.split(features, targets, None)?;
    let index_split = splits.pop().ok_or(SKError::EmptyInput {
        operation: "sk_train_test_split",
    })?;

    let (train_features, train_targets) = sk_gather_rows(features, targets, index_split.train())?;
    let (test_features, test_targets) = sk_gather_rows(features, targets, index_split.test())?;

    Ok(SKTrainTestSplit::new(
        train_features,
        test_features,
        train_targets,
        test_targets,
        index_split.train().to_vec(),
        index_split.test().to_vec(),
    ))
}
