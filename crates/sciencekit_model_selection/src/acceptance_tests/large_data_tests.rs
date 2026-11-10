//! Acceptance on large data: folds stay balanced and exhaustive.

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKDataView, SKTargetView};

use super::fixtures::{contiguous_groups, cyclic_labels, dense_design};
use crate::SKSplitter;
use crate::split::group::SKGroupKFold;
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;
use crate::train_test_split::{SKTrainTestSplitter, sk_train_test_split};

#[test]
fn acceptance_large_kfold_covers_ten_thousand_rows_exactly_once() {
    let features = dense_design(10_000, 8);
    let splitter = SKKFold::build()
        .number_of_splits(5)
        .shuffle(true)
        .random_seed(77)
        .build()
        .unwrap();

    let splits = splitter
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();

    let mut tested = vec![0_usize; 10_000];
    for split in &splits {
        for &row in split.test() {
            tested[row] += 1;
        }
    }
    assert!(tested.iter().all(|&count| count == 1));
    let sizes: Vec<usize> = splits.iter().map(|split| split.number_of_test_rows()).collect();
    assert!(sizes.iter().all(|&size| size == 2_000));
}

#[test]
fn acceptance_large_group_kfold_never_leaks_a_group() {
    let features = dense_design(10_000, 4);
    let groups = contiguous_groups(10_000, 200);
    let splitter = SKGroupKFold::build()
        .number_of_splits(10)
        .random_seed(12)
        .build()
        .unwrap();

    let splits = splitter
        .split(SKDataView::Dense(features.view()), None, Some(groups.view()))
        .unwrap();

    assert_eq!(splits.len(), 10);
    for split in &splits {
        let test_groups: std::collections::HashSet<i64> =
            split.test().iter().map(|&row| groups[row]).collect();
        for &row in split.train() {
            assert!(!test_groups.contains(&groups[row]));
        }
    }
}

#[test]
fn acceptance_large_stratified_kfold_preserves_proportions() {
    let features = dense_design(9_000, 4);
    let labels = cyclic_labels(9_000, 3);
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(5)
        .shuffle(true)
        .random_seed(5)
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
        for class in 0_i64..3 {
            let count = split.test().iter().filter(|&&row| labels[row] == class).count();
            assert!((599..=601).contains(&count), "class {class} saw {count} rows");
        }
    }
}

#[test]
fn acceptance_large_stratified_split_rejects_a_rare_class() {
    let features = dense_design(9_000, 4);
    let mut labels = cyclic_labels(9_000, 3);
    for row in 0..1 {
        labels[row] = 99;
    }
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(5)
        .build()
        .unwrap();

    let error = splitter
        .split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(labels.view())),
            None,
        )
        .expect_err("a singleton class cannot fill five folds");
    assert!(matches!(
        error,
        sciencekit_common::SKError::InvalidHyperparameter { .. }
    ));
}

#[test]
fn acceptance_large_train_test_split_honours_the_requested_fraction() {
    let features = dense_design(10_000, 8);
    let targets = Array1::from_shape_fn(10_000, |row| row as f64);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.2)
        .random_seed(31)
        .build()
        .unwrap();

    let split = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Continuous(targets.view())),
    )
    .unwrap();

    assert_eq!(split.number_of_test_rows(), 2_000);
    assert_eq!(split.number_of_train_rows(), 8_000);
}
