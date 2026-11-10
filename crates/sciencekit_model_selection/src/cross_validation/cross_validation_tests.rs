//! Cross-validation fold count, parallelism and metric production tests
//! (spec `model-selection-splits`, requirement `Cross-validation scoring`).

use ndarray::{Array1, Array2};
use rayon::ThreadPoolBuilder;
use sciencekit_common::{SKBuilder, SKDataView, SKTargetView};
use sciencekit_metrics::scorers::{SKAccuracyScorer, SKMeanSquaredErrorScorer};
use sprs::TriMat;

use super::mock_estimators::{SKMockMajorityClassifier, SKMockMeanRegressor};
use super::{sk_cross_val_score_classifier, sk_cross_val_score_regressor};
use crate::split::kfold::SKKFold;

fn regression_data(number_of_rows: usize) -> (Array2<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((number_of_rows, 2), |(row, column)| {
        (row + column) as f64
    });
    let targets = Array1::from_shape_fn(number_of_rows, |row| row as f64);
    (features, targets)
}

#[test]
fn cross_validation_regressor_returns_one_finite_score_per_fold() {
    let (features, targets) = regression_data(40);
    let splitter = SKKFold::build().number_of_splits(4).build().unwrap();

    let scores = sk_cross_val_score_regressor(
        &SKMockMeanRegressor::new(),
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        &SKMeanSquaredErrorScorer,
    )
    .unwrap();

    assert_eq!(scores.len(), 4);
    assert!(scores.iter().all(|score| score.is_finite() && *score >= 0.0));
}

#[test]
fn cross_validation_classifier_returns_one_score_per_fold() {
    let features = Array2::from_shape_fn((60, 2), |(row, column)| (row + column) as f64);
    let labels = Array1::from_shape_fn(60, |row| (row % 2) as i64);
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();

    let scores = sk_cross_val_score_classifier(
        &SKMockMajorityClassifier::new(),
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Integer(labels.view()),
        &SKAccuracyScorer,
    )
    .unwrap();

    assert_eq!(scores.len(), 5);
    assert!(
        scores
            .iter()
            .all(|score| *score >= 0.0 && *score <= 1.0)
    );
}

#[test]
fn cross_validation_parallel_scores_equal_sequential_scores() {
    let (features, targets) = regression_data(40);
    let splitter = SKKFold::build()
        .number_of_splits(4)
        .shuffle(true)
        .random_seed(19)
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

#[test]
fn cross_validation_dense_and_sparse_inputs_score_identically() {
    let (features, targets) = regression_data(24);
    let mut triplets = TriMat::new((24, 2));
    for (row, column) in features.indexed_iter() {
        triplets.add_triplet(row.0, row.1, *column);
    }
    let sparse = triplets.to_csr();
    let splitter = SKKFold::build().number_of_splits(3).build().unwrap();
    let estimator = SKMockMeanRegressor::new();

    let dense_scores = sk_cross_val_score_regressor(
        &estimator,
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        &SKMeanSquaredErrorScorer,
    )
    .unwrap();
    let sparse_scores = sk_cross_val_score_regressor(
        &estimator,
        &splitter,
        SKDataView::Sparse(sparse.view()),
        SKTargetView::Continuous(targets.view()),
        &SKMeanSquaredErrorScorer,
    )
    .unwrap();

    assert_eq!(dense_scores, sparse_scores);
}

#[test]
fn cross_validation_zero_folds_is_rejected_by_the_splitter() {
    let splitter = SKKFold::build().number_of_splits(2).build().unwrap();
    let empty = Array2::<f64>::zeros((0, 2));
    let empty_targets = Array1::<f64>::zeros(0);

    let error = sk_cross_val_score_regressor(
        &SKMockMeanRegressor::new(),
        &splitter,
        SKDataView::Dense(empty.view()),
        SKTargetView::Continuous(empty_targets.view()),
        &SKMeanSquaredErrorScorer,
    )
    .expect_err("an empty input has no folds");

    assert!(matches!(error, sciencekit_common::SKError::EmptyInput { .. }));
}
