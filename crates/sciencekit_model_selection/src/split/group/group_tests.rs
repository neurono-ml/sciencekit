//! Group K-fold integrity and validation tests (spec `model-selection-splits`,
//! requirement `Group K-fold`).

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError};

use crate::SKSplitter;
use crate::split::group::SKGroupKFold;

fn group_labels() -> Array1<i64> {
    Array1::from_vec(vec![0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
}

#[test]
fn group_never_splits_a_group_across_train_and_test() {
    let data = Array2::<f64>::zeros((10, 2));
    let groups = group_labels();
    let splitter = SKGroupKFold::build().build().unwrap();

    let splits = splitter
        .split(SKDataView::Dense(data.view()), None, Some(groups.view()))
        .unwrap();

    for split in &splits {
        let test_group = groups[split.test()[0]];
        for &row in split.test() {
            assert_eq!(groups[row], test_group);
        }
        for &row in split.train() {
            assert_ne!(
                groups[row], test_group,
                "group {test_group} leaked to train"
            );
        }
    }
}

#[test]
fn group_defaults_to_the_number_of_distinct_groups() {
    let data = Array2::<f64>::zeros((10, 2));
    let groups = group_labels();
    let splitter = SKGroupKFold::build().build().unwrap();

    let splits = splitter
        .split(SKDataView::Dense(data.view()), None, Some(groups.view()))
        .unwrap();

    assert_eq!(splits.len(), 4);
    assert_eq!(splitter.number_of_splits(), None);
}

#[test]
fn group_test_partitions_are_disjoint_and_exhaustive() {
    let data = Array2::<f64>::zeros((10, 2));
    let groups = group_labels();
    let splitter = SKGroupKFold::build()
        .number_of_splits(2)
        .random_seed(17)
        .build()
        .unwrap();

    let splits = splitter
        .split(SKDataView::Dense(data.view()), None, Some(groups.view()))
        .unwrap();

    assert_eq!(splits.len(), 2);
    let mut tested = HashSet::new();
    for split in &splits {
        for &row in split.test() {
            assert!(tested.insert(row));
        }
    }
    assert_eq!(tested.len(), 10);
}

#[test]
fn group_shuffle_is_seeded() {
    let data = Array2::<f64>::zeros((10, 2));
    let groups = group_labels();
    let splitter = SKGroupKFold::build().random_seed(5).build().unwrap();

    let first = splitter
        .split(SKDataView::Dense(data.view()), None, Some(groups.view()))
        .unwrap();
    let second = splitter
        .split(SKDataView::Dense(data.view()), None, Some(groups.view()))
        .unwrap();

    assert_eq!(first, second);
}

#[test]
fn group_rejects_a_split_count_larger_than_the_group_count() {
    let data = Array2::<f64>::zeros((10, 2));
    let groups = group_labels();
    let splitter = SKGroupKFold::build().number_of_splits(5).build().unwrap();

    let error = splitter
        .split(SKDataView::Dense(data.view()), None, Some(groups.view()))
        .expect_err("five splits cannot fill from four groups");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        }
    ));
}

#[test]
fn group_rejects_a_group_length_mismatch() {
    let data = Array2::<f64>::zeros((10, 2));
    let groups = Array1::from_vec(vec![0_i64, 1, 2]);
    let splitter = SKGroupKFold::build().build().unwrap();

    let error = splitter
        .split(SKDataView::Dense(data.view()), None, Some(groups.view()))
        .expect_err("three group labels cannot describe ten rows");

    assert!(matches!(
        error,
        SKError::ShapeMismatch {
            expected,
            found,
        } if expected == vec![10] && found == vec![3]
    ));
}

#[test]
fn group_rejects_missing_group_labels() {
    let data = Array2::<f64>::zeros((10, 2));
    let splitter = SKGroupKFold::build().build().unwrap();

    let error = splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .expect_err("group K-fold needs group labels");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "groups", .. }
    ));
}
