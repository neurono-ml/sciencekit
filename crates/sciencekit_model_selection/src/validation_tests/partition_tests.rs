//! Validation of empty inputs, fractions, counts, cardinality and lengths.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKTargetView};

use crate::SKSplitter;
use crate::split::group::SKGroupKFold;
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;
use crate::train_test_split::{SKTrainTestSplitter, sk_train_test_split};

fn finite_features(number_of_rows: usize) -> Array2<f64> {
    Array2::zeros((number_of_rows, 2))
}

#[test]
fn validation_rejects_empty_input_from_every_splitter() {
    let features = finite_features(0);
    let empty_targets = Array1::<i64>::zeros(0);
    let empty_groups = Array1::<i64>::zeros(0);

    let kfold = SKKFold::build().number_of_splits(2).build().unwrap();
    assert!(matches!(
        kfold.split(SKDataView::Dense(features.view()), None, None),
        Err(SKError::EmptyInput { .. })
    ));

    let stratified = SKStratifiedKFold::build()
        .number_of_splits(2)
        .build()
        .unwrap();
    assert!(matches!(
        stratified.split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(empty_targets.view())),
            None,
        ),
        Err(SKError::EmptyInput { .. })
    ));

    let group = SKGroupKFold::build().build().unwrap();
    assert!(matches!(
        group.split(
            SKDataView::Dense(features.view()),
            None,
            Some(empty_groups.view()),
        ),
        Err(SKError::EmptyInput { .. })
    ));

    let train_test = SKTrainTestSplitter::build().build().unwrap();
    assert!(matches!(
        sk_train_test_split(&train_test, SKDataView::Dense(features.view()), None),
        Err(SKError::EmptyInput { .. })
    ));
}

#[test]
fn validation_rejects_invalid_fractions() {
    for fraction in [0.0, 1.0, -0.25, 1.5, f64::NAN, f64::INFINITY] {
        let error = SKTrainTestSplitter::build()
            .test_fraction(fraction)
            .build()
            .unwrap_err();
        assert!(
            matches!(
                error,
                SKError::InvalidHyperparameter {
                    name: "test_size",
                    ..
                }
            ),
            "fraction {fraction} must be rejected"
        );
    }
}

#[test]
fn validation_rejects_too_many_splits() {
    let features = finite_features(4);
    let kfold = SKKFold::build().number_of_splits(5).build().unwrap();
    assert!(matches!(
        kfold.split(SKDataView::Dense(features.view()), None, None),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        })
    ));

    let groups = Array1::from_vec(vec![0_i64, 0, 1, 1]);
    let group = SKGroupKFold::build().number_of_splits(3).build().unwrap();
    assert!(matches!(
        group.split(
            SKDataView::Dense(features.view()),
            None,
            Some(groups.view()),
        ),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        })
    ));
}

#[test]
fn validation_rejects_a_rare_class_and_a_small_group_count() {
    let features = finite_features(4);
    let targets = Array1::from_vec(vec![0_i64, 0, 0, 1]);
    let stratified = SKStratifiedKFold::build()
        .number_of_splits(2)
        .build()
        .unwrap();
    assert!(matches!(
        stratified.split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(targets.view())),
            None,
        ),
        Err(SKError::InvalidHyperparameter {
            name: "targets",
            ..
        })
    ));

    let groups = Array1::from_vec(vec![0_i64, 0, 1, 1]);
    let group = SKGroupKFold::build().number_of_splits(3).build().unwrap();
    assert!(matches!(
        group.split(
            SKDataView::Dense(features.view()),
            None,
            Some(groups.view()),
        ),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        })
    ));
}

#[test]
fn validation_rejects_length_mismatches() {
    let features = finite_features(5);

    let short_groups = Array1::from_vec(vec![0_i64, 1, 2]);
    let group = SKGroupKFold::build().build().unwrap();
    assert!(matches!(
        group.split(
            SKDataView::Dense(features.view()),
            None,
            Some(short_groups.view()),
        ),
        Err(SKError::ShapeMismatch {
            expected,
            found,
        }) if expected == vec![5] && found == vec![3]
    ));

    let short_targets = Array1::from_vec(vec![0_i64, 1]);
    let train_test = SKTrainTestSplitter::build().build().unwrap();
    assert!(matches!(
        sk_train_test_split(
            &train_test,
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(short_targets.view())),
        ),
        Err(SKError::ShapeMismatch {
            expected,
            found,
        }) if expected == vec![5] && found == vec![2]
    ));
}

#[test]
fn validation_rejects_a_split_count_below_two() {
    assert!(matches!(
        SKKFold::build().number_of_splits(1).build(),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        })
    ));
    assert!(matches!(
        SKStratifiedKFold::build().number_of_splits(1).build(),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        })
    ));
    assert!(matches!(
        SKGroupKFold::build().number_of_splits(1).build(),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        })
    ));
}
