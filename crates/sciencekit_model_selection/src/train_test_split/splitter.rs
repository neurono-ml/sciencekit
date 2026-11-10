//! The train/test splitter.

use ndarray::ArrayView1;
use sciencekit_common::{
    SKDataView, SKError, SKExecutionMode, SKFloat, SKRandomGenerator, SKTargetView,
    SK_RANDOM_DOMAIN_FOLD, sk_derive_seed,
};

use super::builder::SKTrainTestSplitBuilder;
use super::configuration::SKTrainTestSplitConfiguration;
use super::partition_size::SKPartitionSize;
use super::stratification::sk_allocate_class_counts;
use crate::{SKIndexSplit, SKSplitter};

use crate::split::validation::{
    sk_class_groups, sk_number_of_rows, sk_validate_non_empty,
};

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
    configuration: SKTrainTestSplitConfiguration,
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

    pub(super) fn from_configuration(configuration: SKTrainTestSplitConfiguration) -> Self {
        Self { configuration }
    }

    fn resolved_sizes(&self, number_of_rows: usize) -> Result<(usize, usize), SKError> {
        let default_fraction = SKPartitionSize::Fraction(0.25);
        let (number_of_test_rows, number_of_train_rows) =
            match (self.configuration.test_size, self.configuration.train_size) {
                (None, None) => sk_resolve_pair(default_fraction, number_of_rows),
                (Some(test_size), None) => {
                    let test = sk_size_to_count(test_size, number_of_rows, true)?;
                    (test, number_of_rows - test)
                }
                (None, Some(train_size)) => {
                    let train = sk_size_to_count(train_size, number_of_rows, false)?;
                    (number_of_rows - train, train)
                }
                (Some(test_size), Some(train_size)) => (
                    sk_size_to_count(test_size, number_of_rows, true)?,
                    sk_size_to_count(train_size, number_of_rows, false)?,
                ),
            };

        if number_of_test_rows + number_of_train_rows > number_of_rows {
            return Err(SKError::InvalidHyperparameter {
                name: "test_size",
                reason: "the test and train partitions exceed the available rows".to_owned(),
            });
        }
        if number_of_test_rows == 0 || number_of_train_rows == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "test_size",
                reason: "both partitions must contain at least one row".to_owned(),
            });
        }
        Ok((number_of_train_rows, number_of_test_rows))
    }

    fn split_indices<F: SKFloat>(
        &self,
        features: SKDataView<'_, F>,
        targets: Option<SKTargetView<'_, F>>,
    ) -> Result<SKIndexSplit, SKError> {
        let number_of_rows = sk_number_of_rows(&features);
        sk_validate_non_empty(number_of_rows, "sk_train_test_split")?;
        if let Some(targets) = targets {
            if targets.len() != number_of_rows {
                return Err(SKError::ShapeMismatch {
                    expected: vec![number_of_rows],
                    found: vec![targets.len()],
                });
            }
        }

        let (number_of_train_rows, number_of_test_rows) =
            self.resolved_sizes(number_of_rows)?;
        if self.configuration.stratify {
            return self.stratified_indices(
                number_of_rows,
                number_of_train_rows,
                number_of_test_rows,
                targets,
            );
        }

        let permutation = if self.configuration.shuffle {
            crate::sk_shuffle_indices(self.configuration.random_seed, number_of_rows)
        } else {
            (0..number_of_rows).collect()
        };
        let test = permutation[..number_of_test_rows].to_vec();
        let train = permutation
            [number_of_test_rows..number_of_test_rows + number_of_train_rows]
            .to_vec();
        Ok(SKIndexSplit::new(train, test))
    }

    fn stratified_indices<F: SKFloat>(
        &self,
        number_of_rows: usize,
        number_of_train_rows: usize,
        number_of_test_rows: usize,
        targets: Option<SKTargetView<'_, F>>,
    ) -> Result<SKIndexSplit, SKError> {
        let class_groups = sk_class_groups(targets)?;
        let class_sizes: Vec<usize> = class_groups.iter().map(Vec::len).collect();
        let test_counts =
            sk_allocate_class_counts(&class_sizes, number_of_test_rows, number_of_rows);
        let remaining_sizes: Vec<usize> = class_sizes
            .iter()
            .zip(&test_counts)
            .map(|(size, test)| size - test)
            .collect();
        let remaining_rows: usize = remaining_sizes.iter().sum();
        let train_counts =
            sk_allocate_class_counts(&remaining_sizes, number_of_train_rows, remaining_rows);

        let mut train = Vec::with_capacity(number_of_train_rows);
        let mut test = Vec::with_capacity(number_of_test_rows);
        for (class_index, members) in class_groups.iter().enumerate() {
            let mut ordered = members.clone();
            if self.configuration.shuffle {
                let seed = sk_derive_seed(
                    self.configuration.random_seed,
                    SK_RANDOM_DOMAIN_FOLD,
                    class_index as u64,
                );
                SKRandomGenerator::new(seed).shuffle(&mut ordered);
            }
            let test_count = test_counts[class_index];
            let train_count = train_counts[class_index];
            test.extend_from_slice(&ordered[..test_count]);
            train.extend_from_slice(&ordered[test_count..test_count + train_count]);
        }
        Ok(SKIndexSplit::new(train, test))
    }
}

fn sk_resolve_pair(size: SKPartitionSize, number_of_rows: usize) -> (usize, usize) {
    let test = sk_size_to_count(size, number_of_rows, true)
        .expect("the default fraction is valid");
    (test, number_of_rows - test)
}

fn sk_size_to_count(
    size: SKPartitionSize,
    number_of_rows: usize,
    round_up: bool,
) -> Result<usize, SKError> {
    match size {
        SKPartitionSize::Count(count) => Ok(count),
        SKPartitionSize::Fraction(fraction) => {
            let exact = fraction * number_of_rows as f64;
            let count = if round_up {
                exact.ceil() as usize
            } else {
                exact.floor() as usize
            };
            Ok(count)
        }
    }
}

impl<F: SKFloat> SKSplitter<F> for SKTrainTestSplitter {
    fn split(
        &self,
        features: SKDataView<'_, F>,
        targets: Option<SKTargetView<'_, F>>,
        _groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError> {
        Ok(vec![self.split_indices(features, targets)?])
    }
}
