//! Gathering of train and test features and targets.

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKDataView, SKFeatureBlock, SKTargetView};
use sprs::{CsMat, TriMat};

use super::fixtures::dense_features;
use crate::SKTargetPartition;
use crate::train_test_split::{SKTrainTestSplitter, sk_train_test_split};

#[test]
fn train_test_split_gathers_matching_features_and_targets() {
    let features = dense_features(6);
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

    let expected_train =
        ndarray::Array2::from_shape_fn((4, 2), |(row, column)| ((row + 2) * 2 + column) as f64);
    let SKFeatureBlock::Dense(train) = split.train_features() else {
        panic!("a dense input must stay dense");
    };
    assert_eq!(train, &expected_train);
    let Some(SKTargetPartition::Integer(train_targets)) = split.train_targets() else {
        panic!("integer targets must stay integer");
    };
    assert_eq!(train_targets, &Array1::from_vec(vec![2, 3, 4, 5]));
}

#[test]
fn train_test_split_sparse_gathering_never_densifies() {
    let mut triplets = TriMat::new((6, 3));
    triplets.add_triplet(0, 0, 1.0_f64);
    triplets.add_triplet(4, 2, 2.0);
    let sparse: CsMat<f64> = triplets.to_csr();
    let splitter = SKTrainTestSplitter::build()
        .test_count(2)
        .shuffle(false)
        .build()
        .unwrap();

    let split = sk_train_test_split(&splitter, SKDataView::Sparse(sparse.view()), None).unwrap();

    let SKFeatureBlock::Sparse(train) = split.train_features() else {
        panic!("a sparse input must stay sparse");
    };
    let SKFeatureBlock::Sparse(test) = split.test_features() else {
        panic!("a sparse input must stay sparse");
    };
    assert_eq!((train.rows(), train.cols()), (4, 3));
    assert_eq!((test.rows(), test.cols()), (2, 3));
    assert_eq!(train.data().len(), 1);
}
