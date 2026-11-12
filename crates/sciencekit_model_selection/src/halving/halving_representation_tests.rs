//! TDD tests for dense/sparse/mixed representation parity and upfront input validation
//! (spec `halving-grid-search` and spec `data-view-boundary`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sprs::{CsMat, TriMat};

use crate::grid_search::mock_estimators::{
    SKMockConstantRegressor, SKMockSparseOnlyRegressor, SKMockThresholdClassifier,
};
use crate::halving::{SKHalvingGridSearchClassifier, SKHalvingGridSearchRegressor};
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;

fn sparse_from_dense(features: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new(features.dim());
    features
        .indexed_iter()
        .filter(|(_, value)| **value != 0.0)
        .for_each(|((row, col), value)| triplets.add_triplet(row, col, *value));
    triplets.to_csr()
}

#[test]
fn halving_concurrency_representation_dense_sparse_and_mixed_match_without_densification() {
    let features = Array2::from_shape_fn((54, 3), |(row, col)| {
        if (row + col) % 2 == 0 {
            row as f64
        } else {
            0.0
        }
    });
    let sparse_features = sparse_from_dense(&features);
    let labels = Array1::from_iter((0..54).map(|row| i64::from(row >= 27)));

    let classifier = SKHalvingGridSearchClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(8.0),
            SKMockThresholdClassifier::new(27.0),
            SKMockThresholdClassifier::new(44.0),
        ])
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(3)
                .build()
                .unwrap(),
        )
        .factor(3)
        .minimum_resource(18)
        .random_seed(9)
        .build()
        .unwrap();

    let dense_model = classifier.fit(features.view(), labels.view()).unwrap();
    let sparse_model = classifier
        .fit(sparse_features.view(), labels.view())
        .unwrap();

    assert_eq!(
        dense_model.elimination_trace(),
        sparse_model.elimination_trace()
    );
    // Mixed representation predictions: dense-fitted model on sparse CSR and vice versa.
    assert_eq!(
        dense_model.predict_labels(sparse_features.view()).unwrap(),
        sparse_model.predict_labels(features.view()).unwrap()
    );

    // Sparse-only estimator proves neither round subsampling nor final refit densifies CSR.
    let targets = Array1::from_iter((0..54).map(|row| (row % 5) as f64));
    let sparse_only = SKHalvingGridSearchRegressor::build()
        .candidates(vec![SKMockSparseOnlyRegressor, SKMockSparseOnlyRegressor])
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .factor(2)
        .minimum_resource(18)
        .build()
        .unwrap();
    let sparse_reg_model = sparse_only
        .fit(sparse_features.view(), targets.view())
        .unwrap();
    assert!(sparse_reg_model.best_model().unwrap().sparse_training());
    assert_eq!(sparse_reg_model.best_model().unwrap().training_rows(), 54);
    assert_eq!(
        sparse_reg_model
            .predict(sparse_features.view())
            .unwrap()
            .len(),
        54
    );
}

#[test]
fn halving_concurrency_rejects_non_finite_rows_even_outside_sampled_budget() {
    let mut features = Array2::zeros((40, 2));
    features[[39, 1]] = f64::NAN;
    let targets = Array1::<f64>::zeros(40);

    let search = SKHalvingGridSearchRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(1.0),
        ])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .minimum_resource(6)
        .maximum_resource(6)
        .refit(false)
        .random_seed(0)
        .build()
        .unwrap();

    let Err(error) = search.fit(features.view(), targets.view()) else {
        panic!("non-finite input outside the sampled round subset must still be rejected");
    };
    assert!(matches!(error, SKError::NonFiniteValue { position: 79 }));
}
