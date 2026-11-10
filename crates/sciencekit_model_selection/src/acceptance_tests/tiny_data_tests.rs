//! Acceptance on tiny data: every splitter works on a handful of rows.

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKTargetView};

use crate::SKSplitter;
use crate::split::group::SKGroupKFold;
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;
use crate::train_test_split::{SKTrainTestSplitter, sk_train_test_split};

#[test]
fn acceptance_tiny_kfold_partitions_six_rows_into_three_folds() {
    let features = Array2::<f64>::zeros((6, 2));
    let splitter = SKKFold::build().number_of_splits(3).build().unwrap();

    let splits = splitter
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();

    assert_eq!(splits.len(), 3);
    let tested: HashSet<usize> = splits.iter().flat_map(|split| split.test()).copied().collect();
    assert_eq!(tested, (0..6).collect());
}

#[test]
fn acceptance_tiny_stratified_kfold_sees_both_classes_in_every_fold() {
    let features = Array2::<f64>::zeros((6, 2));
    let labels = Array1::from_vec(vec![0_i64, 0, 0, 1, 1, 1]);
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(3)
        .build()
        .unwrap();

    let splits = splitter
        .split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(labels.view())),
            None,
        )
        .unwrap();

    for split in &splits {
        let zeros = split.test().iter().filter(|&&row| labels[row] == 0).count();
        let ones = split.test().iter().filter(|&&row| labels[row] == 1).count();
        assert_eq!(zeros, 1);
        assert_eq!(ones, 1);
    }
}

#[test]
fn acceptance_tiny_group_kfold_keeps_each_group_together() {
    let features = Array2::<f64>::zeros((6, 2));
    let groups = Array1::from_vec(vec![10_i64, 10, 20, 20, 30, 30]);
    let splitter = SKGroupKFold::build().build().unwrap();

    let splits = splitter
        .split(SKDataView::Dense(features.view()), None, Some(groups.view()))
        .unwrap();

    assert_eq!(splits.len(), 3);
    for split in &splits {
        let test_group = groups[split.test()[0]];
        assert_eq!(split.test().len(), 2);
        assert!(split.test().iter().all(|&row| groups[row] == test_group));
        assert!(split.train().iter().all(|&row| groups[row] != test_group));
    }
}

#[test]
fn acceptance_tiny_train_test_split_gathers_both_partitions() {
    let features = Array2::from_shape_fn((6, 2), |(row, column)| (row + column) as f64);
    let targets = Array1::from_vec(vec![0_i64, 1, 2, 3, 4, 5]);
    let splitter = SKTrainTestSplitter::build()
        .test_count(2)
        .shuffle(false)
        .build()
        .unwrap();

    let split = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
    )
    .unwrap();

    assert_eq!(split.number_of_test_rows(), 2);
    assert_eq!(split.number_of_train_rows(), 4);
}

#[test]
fn acceptance_tiny_index_splits_address_only_original_rows_and_are_disjoint() {
    let features = Array2::<f64>::zeros((6, 2));
    let splitter = SKKFold::build()
        .number_of_splits(3)
        .shuffle(true)
        .random_seed(3)
        .build()
        .unwrap();

    for split in splitter
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap()
    {
        let train: HashSet<usize> = split.train().iter().copied().collect();
        let test: HashSet<usize> = split.test().iter().copied().collect();
        assert!(train.iter().chain(&test).all(|&row| row < 6));
        assert!(train.is_disjoint(&test));
    }
}
