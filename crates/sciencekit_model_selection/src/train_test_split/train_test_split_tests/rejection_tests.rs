//! Rejection of invalid fractions, counts, inputs and lengths.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKTargetView};

use super::fixtures::dense_features;
use crate::train_test_split::{SKPartitionSize, SKTrainTestSplitter, sk_train_test_split};

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
    let splitter = SKTrainTestSplitter::build().test_count(10).build().unwrap();

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

/// An explicit count larger than the input is rejected, not an overflow.
#[test]
fn train_test_split_rejects_a_count_larger_than_the_input() {
    let features = ndarray::Array2::<f64>::zeros((10, 1));
    for splitter in [
        SKTrainTestSplitter::build()
            .test_size(SKPartitionSize::Count(15))
            .build()
            .unwrap(),
        SKTrainTestSplitter::build()
            .train_size(SKPartitionSize::Count(15))
            .build()
            .unwrap(),
    ] {
        let error =
            sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None).unwrap_err();
        assert!(matches!(
            error,
            sciencekit_common::SKError::InvalidHyperparameter { .. }
        ));
    }
}
