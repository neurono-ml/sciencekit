//! Dense/sparse parity without densification (spec `grid-search`, requirement
//! `Representation parity`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sprs::{CsMat, TriMat};

use crate::grid_search::mock_estimators::{
    SKMockConstantRegressor, SKMockSparseOnlyRegressor, SKMockThresholdClassifier,
};
use crate::grid_search::{SKGridSearchCVClassifier, SKGridSearchCVRegressor};
use crate::split::kfold::SKKFold;

fn dense_fixtures() -> (Array2<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((20, 3), |(row, column)| ((row * 3 + column) % 11) as f64);
    let targets = Array1::from_iter((0..20).map(|row| (row % 5) as f64));
    (features, targets)
}

fn sparse(features: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new(features.dim());
    for ((row, column), value) in features.indexed_iter() {
        if *value != 0.0 {
            triplets.add_triplet(row, column, *value);
        }
    }
    triplets.to_csr()
}

#[test]
fn grid_representation_sparse_regression_matches_dense_results() {
    let (features, targets) = dense_fixtures();
    let sparse_features = sparse(&features);
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(2.0),
            SKMockConstantRegressor::new(4.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap();

    let dense_model = estimator.fit(features.view(), targets.view()).unwrap();
    let sparse_model = estimator
        .fit(sparse_features.view(), targets.view())
        .unwrap();

    assert_eq!(dense_model.fold_scores(), sparse_model.fold_scores());
    assert_eq!(dense_model.mean_scores(), sparse_model.mean_scores());
    assert!(!dense_model.best_model().unwrap().sparse_training());
    assert!(sparse_model.best_model().unwrap().sparse_training());
    assert_eq!(
        dense_model.best_candidate_index(),
        sparse_model.best_candidate_index()
    );
}

#[test]
fn grid_representation_sparse_search_never_densifies_a_fold() {
    let (features, targets) = dense_fixtures();
    let sparse_features = sparse(&features);
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![SKMockSparseOnlyRegressor])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap();

    let model = estimator
        .fit(sparse_features.view(), targets.view())
        .unwrap();
    let responses = model.predict(sparse_features.view()).unwrap();
    assert_eq!(responses.len(), 20);
    assert_eq!(model.best_model().unwrap().training_rows(), 20);
    assert!(model.best_model().unwrap().sparse_training());
}

#[test]
fn grid_representation_sparse_classification_predicts_like_dense() {
    let (features, _) = dense_fixtures();
    let sparse_features = sparse(&features);
    let labels = Array1::from_iter((0..20).map(|row| i64::from(row >= 10)));
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(3.0),
            SKMockThresholdClassifier::new(6.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap();

    let dense_model = estimator.fit(features.view(), labels.view()).unwrap();
    let sparse_model = estimator
        .fit(sparse_features.view(), labels.view())
        .unwrap();

    let dense_labels = dense_model.predict_labels(features.view()).unwrap();
    let sparse_labels = sparse_model.predict_labels(sparse_features.view()).unwrap();
    assert_eq!(dense_labels, sparse_labels);
}
