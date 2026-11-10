//! Plain K-fold partition properties (spec `model-selection-splits`,
//! requirement `K-fold`).

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKTargetView};

use crate::SKSplitter;
use crate::split::kfold::SKKFold;

fn features(number_of_rows: usize) -> Array2<f64> {
    Array2::zeros((number_of_rows, 2))
}

#[test]
fn kfold_test_partitions_are_disjoint_and_exhaustive() {
    let data = features(50);
    let kfold = SKKFold::build().number_of_splits(5).build().unwrap();

    let splits = kfold
        .split(SKDataView::Dense(data.view()), None, None)
        .unwrap();

    assert_eq!(splits.len(), 5);
    let mut tested = HashSet::new();
    for split in &splits {
        for &index in split.test() {
            assert!(tested.insert(index), "row {index} is tested more than once");
        }
        assert_eq!(split.number_of_test_rows(), 10);
        assert_eq!(split.number_of_train_rows(), 40);
    }
    assert_eq!(tested.len(), 50);
}

#[test]
fn kfold_without_shuffle_uses_hand_checkable_contiguous_folds() {
    let data = features(10);
    let kfold = SKKFold::build().number_of_splits(3).build().unwrap();

    let splits = kfold
        .split(SKDataView::Dense(data.view()), None, None)
        .unwrap();

    assert_eq!(splits[0].test(), &[0, 1, 2, 3]);
    assert_eq!(splits[1].test(), &[4, 5, 6]);
    assert_eq!(splits[2].test(), &[7, 8, 9]);
    assert_eq!(splits[0].train(), &[4, 5, 6, 7, 8, 9]);
}

#[test]
fn kfold_shuffle_is_seeded_and_reproducible() {
    let data = features(60);
    let splitter = SKKFold::build()
        .number_of_splits(4)
        .shuffle(true)
        .random_seed(13)
        .build()
        .unwrap();

    let first = splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .unwrap();
    let second = splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .unwrap();

    assert_eq!(first, second);
    let shuffled = first
        .iter()
        .flat_map(|split| split.test().iter().copied())
        .collect::<Vec<_>>();
    assert_ne!(shuffled, (0..60).collect::<Vec<_>>());
}

#[test]
fn kfold_train_and_test_never_overlap_and_cover_every_row() {
    let data = features(37);
    let splitter = SKKFold::build()
        .number_of_splits(6)
        .shuffle(true)
        .random_seed(4)
        .build()
        .unwrap();

    for split in splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .unwrap()
    {
        let train: HashSet<usize> = split.train().iter().copied().collect();
        let test: HashSet<usize> = split.test().iter().copied().collect();
        assert!(train.is_disjoint(&test));
        assert_eq!(train.len() + test.len(), 37);
        assert_eq!(train.len(), split.number_of_train_rows());
        assert_eq!(test.len(), split.number_of_test_rows());
    }
}

#[test]
fn kfold_ignores_the_optional_targets_and_groups() {
    let data = features(12);
    let splitter = SKKFold::build().number_of_splits(3).build().unwrap();
    let targets = Array1::from_vec((0..12).map(|value| value % 2).collect());
    let groups = Array1::from_vec((0..12).map(|value| value % 4).collect());

    let plain = splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .unwrap();
    let with_optional = splitter
        .split(
            SKDataView::Dense(data.view()),
            Some(SKTargetView::Integer(targets.view())),
            Some(groups.view()),
        )
        .unwrap();

    assert_eq!(plain, with_optional);
}

#[test]
fn kfold_rejects_too_many_splits() {
    let data = features(4);
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();

    let error = splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .expect_err("five splits cannot cover four rows");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        }
    ));
}

#[test]
fn kfold_rejects_empty_input() {
    let data = features(0);
    let splitter = SKKFold::build().number_of_splits(2).build().unwrap();

    let error = splitter
        .split(SKDataView::Dense(data.view()), None, None)
        .expect_err("an empty input has no folds");

    assert!(matches!(
        error,
        SKError::EmptyInput {
            operation: "SKKFold::split"
        }
    ));
}

#[test]
fn kfold_builder_rejects_fewer_than_two_splits() {
    let error = SKKFold::build().number_of_splits(1).build().unwrap_err();

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        }
    ));
}
