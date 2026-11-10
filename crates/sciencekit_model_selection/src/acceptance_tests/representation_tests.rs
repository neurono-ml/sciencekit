//! Acceptance: dense and sparse inputs produce identical partitions and a
//! sparse input is never densified (spec `data-view-boundary`).

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKDataView, SKFeatureBlock, SKTargetView};

use super::fixtures::{contiguous_groups, cyclic_labels, dense_design, sparse_design};
use crate::SKSplitter;
use crate::split::group::SKGroupKFold;
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;
use crate::train_test_split::{SKTrainTestSplitter, sk_train_test_split};

#[test]
fn acceptance_representation_kfold_splits_match_between_dense_and_sparse() {
    let dense = dense_design(120, 5);
    let sparse = sparse_design(&dense);
    let splitter = SKKFold::build()
        .number_of_splits(4)
        .shuffle(true)
        .random_seed(23)
        .build()
        .unwrap();

    let dense_splits = splitter
        .split(SKDataView::Dense(dense.view()), None, None)
        .unwrap();
    let sparse_splits = splitter
        .split(SKDataView::Sparse(sparse.view()), None, None)
        .unwrap();

    assert_eq!(dense_splits, sparse_splits);
}

#[test]
fn acceptance_representation_stratified_kfold_matches_between_representations() {
    let dense = dense_design(120, 5);
    let sparse = sparse_design(&dense);
    let labels = cyclic_labels(120, 3);
    let splitter = SKStratifiedKFold::build()
        .number_of_splits(3)
        .shuffle(true)
        .random_seed(9)
        .build()
        .unwrap();

    let dense_splits = splitter
        .split(
            SKDataView::Dense(dense.view()),
            Some(SKTargetView::Integer(labels.view())),
            None,
        )
        .unwrap();
    let sparse_splits = splitter
        .split(
            SKDataView::Sparse(sparse.view()),
            Some(SKTargetView::Integer(labels.view())),
            None,
        )
        .unwrap();

    assert_eq!(dense_splits, sparse_splits);
}

#[test]
fn acceptance_representation_group_kfold_matches_between_representations() {
    let dense = dense_design(120, 5);
    let sparse = sparse_design(&dense);
    let groups = contiguous_groups(120, 12);
    let splitter = SKGroupKFold::build().number_of_splits(4).build().unwrap();

    let dense_splits = splitter
        .split(SKDataView::Dense(dense.view()), None, Some(groups.view()))
        .unwrap();
    let sparse_splits = splitter
        .split(SKDataView::Sparse(sparse.view()), None, Some(groups.view()))
        .unwrap();

    assert_eq!(dense_splits, sparse_splits);
}

#[test]
fn acceptance_representation_train_test_split_matches_between_representations() {
    let dense = dense_design(100, 5);
    let sparse = sparse_design(&dense);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.3)
        .random_seed(4)
        .build()
        .unwrap();

    let dense_split =
        sk_train_test_split(&splitter, SKDataView::Dense(dense.view()), None).unwrap();
    let sparse_split =
        sk_train_test_split(&splitter, SKDataView::Sparse(sparse.view()), None).unwrap();

    assert_eq!(dense_split.train_indices(), sparse_split.train_indices());
    assert_eq!(dense_split.test_indices(), sparse_split.test_indices());
}

#[test]
fn acceptance_representation_sparse_output_stays_sparse() {
    let dense = dense_design(40, 4);
    let sparse = sparse_design(&dense);
    let splitter = SKTrainTestSplitter::build()
        .test_count(10)
        .random_seed(2)
        .build()
        .unwrap();

    let split = sk_train_test_split(&splitter, SKDataView::Sparse(sparse.view()), None).unwrap();

    assert!(matches!(split.train_features(), SKFeatureBlock::Sparse(_)));
    assert!(matches!(split.test_features(), SKFeatureBlock::Sparse(_)));
    let stored_before = sparse.data().len();
    let stored_after = match split.train_features() {
        SKFeatureBlock::Sparse(train) => train.data().len(),
        _ => unreachable!("the block is sparse"),
    } + match split.test_features() {
        SKFeatureBlock::Sparse(test) => test.data().len(),
        _ => unreachable!("the block is sparse"),
    };
    assert_eq!(stored_before, stored_after);
}

#[test]
fn acceptance_representation_targets_keep_their_representation() {
    let dense = dense_design(30, 3);
    let sparse = sparse_design(&dense);
    let labels = Array1::from_shape_fn(30, |row| (row % 2) as i64);
    let splitter = SKTrainTestSplitter::build()
        .test_count(6)
        .build()
        .unwrap();

    let split = sk_train_test_split(
        &splitter,
        SKDataView::Sparse(sparse.view()),
        Some(SKTargetView::Integer(labels.view())),
    )
    .unwrap();

    assert!(matches!(
        split.test_targets(),
        Some(crate::SKTargetPartition::Integer(_))
    ));
}
