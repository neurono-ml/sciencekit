//! Train/test split fraction, count, shuffle, gathering and validation tests
//! (spec `model-selection-splits`, requirement `Train/test split`).

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKFeatureBlock, SKTargetView};
use sprs::{CsMat, TriMat};

use super::{SKPartitionSize, SKTrainTestSplitter, sk_train_test_split};

fn dense_features(number_of_rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((number_of_rows, 2), |(row, column)| (row * 2 + column) as f64)
}

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
    let splitter = SKTrainTestSplitter::build()
        .test_count(15)
        .build()
        .unwrap();

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

    let expected_train = Array2::from_shape_fn((4, 2), |(row, column)| {
        ((row + 2) * 2 + column) as f64
    });
    let SKFeatureBlock::Dense(train) = split.train_features() else {
        panic!("a dense input must stay dense");
    };
    assert_eq!(train, &expected_train);
    let Some(crate::SKTargetPartition::Integer(train_targets)) = split.train_targets() else {
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

#[test]
fn train_test_split_rejects_invalid_fractions_and_counts_at_build() {
    for fraction in [0.0, 1.0, -0.5, f64::NAN] {
        let error = SKTrainTestSplitter::build()
            .test_fraction(fraction)
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "test_size",
                ..
            }
        ));
    }
    let error = SKTrainTestSplitter::build()
        .test_count(0)
        .build()
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "test_size",
            ..
        }
    ));
}

#[test]
fn train_test_split_rejects_fractions_that_consume_every_row() {
    let error = SKTrainTestSplitter::build()
        .test_fraction(0.6)
        .train_fraction(0.4)
        .build()
        .unwrap_err();

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "train_size",
            ..
        }
    ));
}

#[test]
fn train_test_split_rejects_a_count_that_leaves_no_rows_for_the_other_side() {
    let features = dense_features(10);
    let splitter = SKTrainTestSplitter::build()
        .test_count(10)
        .build()
        .unwrap();

    let error = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None)
        .expect_err("ten test rows leave no train rows");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "test_size",
            ..
        }
    ));
}

#[test]
fn train_test_split_rejects_empty_input() {
    let features = Array2::<f64>::zeros((0, 2));
    let splitter = SKTrainTestSplitter::build().build().unwrap();

    let error = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None)
        .expect_err("an empty input cannot be split");

    assert!(matches!(
        error,
        SKError::EmptyInput {
            operation: "sk_train_test_split"
        }
    ));
}

#[test]
fn train_test_split_rejects_a_target_length_mismatch() {
    let features = dense_features(10);
    let targets = Array1::from_vec(vec![0_i64, 1, 2]);
    let splitter = SKTrainTestSplitter::build().build().unwrap();

    let error = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
    )
    .expect_err("three targets cannot describe ten rows");

    assert!(matches!(
        error,
        SKError::ShapeMismatch {
            expected,
            found,
        } if expected == vec![10] && found == vec![3]
    ));
}
