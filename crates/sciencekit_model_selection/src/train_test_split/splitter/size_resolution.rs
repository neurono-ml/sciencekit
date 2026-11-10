//! Resolution of the requested train and test row counts.

use sciencekit_common::SKError;

use super::core_implementation::SKTrainTestSplitter;
use crate::train_test_split::partition_size::SKPartitionSize;

impl SKTrainTestSplitter {
    pub(super) fn resolved_sizes(&self, number_of_rows: usize) -> Result<(usize, usize), SKError> {
        let default_fraction = SKPartitionSize::Fraction(0.25);
        let (number_of_test_rows, number_of_train_rows) =
            match (self.configuration.test_size, self.configuration.train_size) {
                (None, None) => sk_resolve_pair(default_fraction, number_of_rows),
                (Some(test_size), None) => {
                    let test = sk_size_to_count(test_size, number_of_rows, true);
                    let train = number_of_rows
                        .checked_sub(test)
                        .ok_or_else(|| sk_count_exceeds_rows("test_size", test, number_of_rows))?;
                    (test, train)
                }
                (None, Some(train_size)) => {
                    let train = sk_size_to_count(train_size, number_of_rows, false);
                    let test = number_of_rows.checked_sub(train).ok_or_else(|| {
                        sk_count_exceeds_rows("train_size", train, number_of_rows)
                    })?;
                    (test, train)
                }
                (Some(test_size), Some(train_size)) => (
                    sk_size_to_count(test_size, number_of_rows, true),
                    sk_size_to_count(train_size, number_of_rows, false),
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
}

fn sk_count_exceeds_rows(name: &'static str, requested: usize, rows: usize) -> SKError {
    SKError::InvalidHyperparameter {
        name,
        reason: format!("{requested} rows were requested but the input has only {rows}"),
    }
}

fn sk_resolve_pair(size: SKPartitionSize, number_of_rows: usize) -> (usize, usize) {
    let test = sk_size_to_count(size, number_of_rows, true);
    (test, number_of_rows - test)
}

fn sk_size_to_count(size: SKPartitionSize, number_of_rows: usize, round_up: bool) -> usize {
    match size {
        SKPartitionSize::Count(count) => count,
        SKPartitionSize::Fraction(fraction) => {
            let exact = fraction * number_of_rows as f64;
            if round_up {
                exact.ceil() as usize
            } else {
                exact.floor() as usize
            }
        }
    }
}
