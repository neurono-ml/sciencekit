//! Acceptance: the same seed always produces the same partitions.

use rayon::ThreadPoolBuilder;
use sciencekit_common::{SKBuilder, SKDataView, SKTargetView};
use sciencekit_metrics::scorers::SKMeanSquaredErrorScorer;

use super::fixtures::{contiguous_groups, cyclic_labels, dense_design};
use crate::SKSplitter;
use crate::cross_validation::mock_estimators::SKMockMeanRegressor;
use crate::cross_validation::sk_cross_val_score_regressor;
use crate::split::group::SKGroupKFold;
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;
use crate::train_test_split::{SKTrainTestSplitter, sk_train_test_split};

#[test]
fn acceptance_determinism_kfold_repeats_identically() {
    let features = dense_design(200, 6);
    let splitter = SKKFold::build()
        .number_of_splits(5)
        .shuffle(true)
        .random_seed(101)
        .build()
        .unwrap();

    let first = splitter
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();
    let second = splitter
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();

    assert_eq!(first, second);
}

#[test]
fn acceptance_determinism_different_seeds_change_the_kfold_order() {
    let features = dense_design(200, 6);
    let first = SKKFold::build()
        .number_of_splits(4)
        .shuffle(true)
        .random_seed(1)
        .build()
        .unwrap()
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();
    let second = SKKFold::build()
        .number_of_splits(4)
        .shuffle(true)
        .random_seed(2)
        .build()
        .unwrap()
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();

    assert_ne!(first[0].test(), second[0].test());
}

#[test]
fn acceptance_determinism_stratified_group_and_train_test_repeat_identically() {
    let features = dense_design(120, 4);
    let labels = cyclic_labels(120, 3);
    let groups = contiguous_groups(120, 12);

    let stratified = SKStratifiedKFold::build()
        .number_of_splits(3)
        .shuffle(true)
        .random_seed(8)
        .build()
        .unwrap();
    let first = stratified
        .split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(labels.view())),
            None,
        )
        .unwrap();
    let second = stratified
        .split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(labels.view())),
            None,
        )
        .unwrap();
    assert_eq!(first, second);

    let group = SKGroupKFold::build()
        .number_of_splits(4)
        .random_seed(8)
        .build()
        .unwrap();
    let first = group
        .split(
            SKDataView::Dense(features.view()),
            None,
            Some(groups.view()),
        )
        .unwrap();
    let second = group
        .split(
            SKDataView::Dense(features.view()),
            None,
            Some(groups.view()),
        )
        .unwrap();
    assert_eq!(first, second);

    let train_test = SKTrainTestSplitter::build()
        .test_fraction(0.25)
        .random_seed(8)
        .build()
        .unwrap();
    let first = sk_train_test_split(&train_test, SKDataView::Dense(features.view()), None).unwrap();
    let second =
        sk_train_test_split(&train_test, SKDataView::Dense(features.view()), None).unwrap();
    assert_eq!(first.train_indices(), second.train_indices());
    assert_eq!(first.test_indices(), second.test_indices());
}

#[test]
fn acceptance_determinism_cross_validation_scores_ignore_the_thread_count() {
    let features = dense_design(80, 4);
    let targets = ndarray::Array1::from_shape_fn(80, |row| row as f64);
    let splitter = SKKFold::build()
        .number_of_splits(5)
        .shuffle(true)
        .random_seed(3)
        .build()
        .unwrap();
    let estimator = SKMockMeanRegressor::new();

    let run = |threads: usize| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            sk_cross_val_score_regressor(
                &estimator,
                &splitter,
                SKDataView::Dense(features.view()),
                SKTargetView::Continuous(targets.view()),
                &SKMeanSquaredErrorScorer,
            )
            .unwrap()
        })
    };

    assert_eq!(run(1), run(4));
}
