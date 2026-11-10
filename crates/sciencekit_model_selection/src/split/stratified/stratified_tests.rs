//! Stratified K-fold proportion and rejection tests (spec
//! `model-selection-splits`, requirement `Stratified K-fold`).

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKTargetView};

use crate::SKSplitter;
use crate::split::stratified::SKStratifiedKFold;

fn balanced_targets() -> Array1<i64> {
    let mut labels = Vec::new();
    labels.extend(std::iter::repeat_n(0_i64, 30));
    labels.extend(std::iter::repeat_n(1_i64, 20));
    labels.extend(std::iter::repeat_n(2_i64, 10));
    Array1::from_vec(labels)
}

fn count_labels(split: &crate::SKIndexSplit, targets: &Array1<i64>) -> [usize; 3] {
    let mut counts = [0; 3];
    for &row in split.test() {
        counts[targets[row] as usize] += 1;
    }
    counts
}

#[test]
fn stratified_every_fold_sees_every_class_in_proportion() {
    let data = Array2::<f64>::zeros((60, 2));
    let targets = balanced_targets();
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(5)
        .build()
        .unwrap();

    let splits = splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Integer(targets.view())),
            None,
        )
        .unwrap();

    assert_eq!(splits.len(), 5);
    for split in &splits {
        assert_eq!(count_labels(split, &targets), [6, 4, 2]);
    }
}

#[test]
fn stratified_test_partitions_are_disjoint_and_exhaustive() {
    let data = Array2::<f64>::zeros((60, 2));
    let targets = balanced_targets();
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(4)
        .shuffle(true)
        .random_seed(21)
        .build()
        .unwrap();

    let mut tested = HashSet::new();
    for split in splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Integer(targets.view())),
            None,
        )
        .unwrap()
    {
        for &row in split.test() {
            assert!(tested.insert(row));
        }
    }
    assert_eq!(tested.len(), 60);
}

#[test]
fn stratified_shuffle_is_seeded() {
    let data = Array2::<f64>::zeros((60, 2));
    let targets = balanced_targets();
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(3)
        .shuffle(true)
        .random_seed(9)
        .build()
        .unwrap();

    let first = splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Integer(targets.view())),
            None,
        )
        .unwrap();
    let second = splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Integer(targets.view())),
            None,
        )
        .unwrap();

    assert_eq!(first, second);
}

#[test]
fn stratified_rejects_a_class_smaller_than_the_split_count() {
    let data = Array2::<f64>::zeros((10, 2));
    let targets = Array1::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 2, 2, 2, 2]);
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(5)
        .build()
        .unwrap();

    let error = splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Integer(targets.view())),
            None,
        )
        .expect_err("a class with two members cannot fill five folds");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "targets",
            ..
        }
    ));
}

#[test]
fn stratified_rejects_continuous_targets() {
    let data = Array2::<f64>::zeros((10, 2));
    let targets = Array1::from_vec((0..10).map(f64::from).collect());
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(2)
        .build()
        .unwrap();

    let error = splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Continuous(targets.view())),
            None,
        )
        .expect_err("continuous targets cannot be stratified");

    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

#[test]
fn stratified_rejects_missing_targets() {
    let data = Array2::<f64>::zeros((10, 2));
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(2)
        .build()
        .unwrap();

    let error = splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .expect_err("stratification needs targets");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "targets",
            ..
        }
    ));
}

#[test]
fn stratified_supports_nominal_targets() {
    let data = Array2::<f64>::zeros((10, 2));
    let labels = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"];
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(2)
        .build()
        .unwrap();

    let splits = splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Nominal(&labels)),
            None,
        )
        .unwrap();

    assert_eq!(splits.len(), 2);
    for split in &splits {
        let first_class = split.test().iter().filter(|&&row| row < 5).count();
        let second_class = split.test().len() - first_class;
        assert_eq!(first_class, second_class);
    }
}
