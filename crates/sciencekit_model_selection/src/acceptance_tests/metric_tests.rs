//! Acceptance: cross-validation produces metrics and the builders implement the
//! shared contracts.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKDataView, SKFloat, SKSupervisedFit, SKTargetView,
};
use sciencekit_metrics::scorers::{SKAccuracyScorer, SKMeanAbsoluteErrorScorer, SKMeanSquaredErrorScorer};

use super::fixtures::{cyclic_labels, dense_design};
use crate::SKSplitter;
use crate::cross_validation::mock_estimators::{SKMockMajorityClassifier, SKMockMeanRegressor};
use crate::cross_validation::{sk_cross_val_score_classifier, sk_cross_val_score_regressor};
use crate::split::group::{SKGroupKFold, SKGroupKFoldBuilder};
use crate::split::kfold::{SKKFold, SKKFoldBuilder};
use crate::split::stratified::{SKStratifiedKFold, SKStratifiedKFoldBuilder};
use crate::train_test_split::{SKTrainTestSplitBuilder, SKTrainTestSplitter};

fn assert_builder<B, M>()
where
    B: SKBuilder<M>,
{
}

fn assert_splitter<F: SKFloat, S: SKSplitter<F>>() {}

fn assert_supervised_fit<F: SKFloat, E: SKSupervisedFit<F>>() {}

#[test]
fn acceptance_metric_builders_and_splitters_implement_the_shared_contracts() {
    assert_builder::<SKKFoldBuilder, SKKFold>();
    assert_builder::<SKStratifiedKFoldBuilder, SKStratifiedKFold>();
    assert_builder::<SKGroupKFoldBuilder, SKGroupKFold>();
    assert_builder::<SKTrainTestSplitBuilder, SKTrainTestSplitter>();

    assert_splitter::<f64, SKKFold>();
    assert_splitter::<f64, SKStratifiedKFold>();
    assert_splitter::<f64, SKGroupKFold>();
    assert_splitter::<f64, SKTrainTestSplitter>();

    assert_supervised_fit::<f64, SKMockMeanRegressor<f64>>();
    assert_supervised_fit::<f64, SKMockMajorityClassifier<f64>>();
}

#[test]
fn acceptance_metric_regression_cross_validation_produces_finite_scores() {
    let features = dense_design(100, 5);
    let targets = Array1::from_shape_fn(100, |row| (row % 17) as f64);
    let splitter = SKKFold::build()
        .number_of_splits(5)
        .shuffle(true)
        .random_seed(15)
        .build()
        .unwrap();

    let mean_squared = sk_cross_val_score_regressor(
        &SKMockMeanRegressor::new(),
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        &SKMeanSquaredErrorScorer,
    )
    .unwrap();
    let mean_absolute = sk_cross_val_score_regressor(
        &SKMockMeanRegressor::new(),
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        &SKMeanAbsoluteErrorScorer,
    )
    .unwrap();

    assert_eq!(mean_squared.len(), 5);
    assert_eq!(mean_absolute.len(), 5);
    assert!(mean_squared.iter().all(|score| score.is_finite() && *score >= 0.0));
    assert!(mean_absolute.iter().all(|score| score.is_finite() && *score >= 0.0));
}

#[test]
fn acceptance_metric_classification_cross_validation_scores_accuracy() {
    let features: Array2<f64> = dense_design(60, 3);
    let labels = cyclic_labels(60, 2);
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
    assert!(scores.iter().all(|score| (*score - 0.5).abs() < 1e-12));
}
