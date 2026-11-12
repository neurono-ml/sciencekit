//! Fraction, count, shuffle and partition-size tests.

use std::collections::HashSet;

use sciencekit_common::{SKBuilder, SKDataView};

use super::fixtures::dense_features;
use crate::train_test_split::{SKPartitionSize, SKTrainTestSplitter, sk_train_test_split};

#[test]
fn train_test_split_default_test_fraction_is_a_quarter() {
    let features = dense_features(100);
    let splitter = SKTrainTestSplitter::build().build().unwrap();

    let split = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap();

    assert_eq!(split.number_of_test_rows(), 25);
    assert_eq!(split.number_of_train_rows(), 75);
}

#[test]
fn train_test_split_explicit_count_is_honoured() {
    let features = dense_features(100);
    let splitter = SKTrainTestSplitter::build().test_count(15).build().unwrap();

    let split = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap();

    assert_eq!(split.number_of_test_rows(), 15);
    assert_eq!(split.number_of_train_rows(), 85);
}

#[test]
fn train_test_split_shuffle_is_reproducible() {
    let features = dense_features(80);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.2)
        .shuffle(true)
        .random_seed(11)
        .build()
        .unwrap();

    let first = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap();
    let second = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap();

    assert_eq!(first.train_indices(), second.train_indices());
    assert_eq!(first.test_indices(), second.test_indices());
    assert_ne!(first.test_indices(), &(0..16).collect::<Vec<_>>());
}

#[test]
fn train_test_split_without_shuffle_takes_the_first_rows_as_test() {
    let features = dense_features(10);
    let splitter = SKTrainTestSplitter::build()
        .test_count(3)
        .shuffle(false)
        .build()
        .unwrap();

    let split = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap();

    assert_eq!(split.test_indices(), &[0, 1, 2]);
    assert_eq!(split.train_indices(), &[3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn train_test_split_both_sizes_leave_surplus_rows_unassigned() {
    let features = dense_features(10);
    let splitter = SKTrainTestSplitter::build()
        .test_size(SKPartitionSize::Count(2))
        .train_size(SKPartitionSize::Count(3))
        .shuffle(false)
        .build()
        .unwrap();

    let split = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap();

    assert_eq!(split.test_indices(), &[0, 1]);
    assert_eq!(split.train_indices(), &[2, 3, 4]);
}

#[test]
fn train_test_split_partitions_are_disjoint_and_cover_the_assigned_rows() {
    let features = dense_features(20);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.3)
        .random_seed(3)
        .build()
        .unwrap();

    let split = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap();
    let train: HashSet<usize> = split.train_indices().iter().copied().collect();
    let test: HashSet<usize> = split.test_indices().iter().copied().collect();

    assert!(train.is_disjoint(&test));
    assert_eq!(train.len(), 14);
    assert_eq!(test.len(), 6);
}
