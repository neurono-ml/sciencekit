//! Dense/sparse representation-parity tests (task 6.2, spec
//! `data-view-boundary`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sprs::CsMat;

use crate::classifier::SKDecisionTreeClassifier;
use crate::regressor::SKDecisionTreeRegressor;

/// The logical fixture: feature 0 and feature 1 both separate the classes,
/// while many zeros are implicit in CSR.
fn classifier_fixtures() -> (ndarray::Array2<f64>, CsMat<f64>, ndarray::Array1<i64>) {
    let dense = array![
        [0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let sparse = CsMat::new(
        (4, 3),
        vec![0, 1, 2, 3, 4],
        vec![1, 1, 0, 0],
        vec![2.0, 1.0, 3.0, 4.0],
    );
    (dense, sparse, array![0_i64, 0, 1, 1])
}

/// Fitting the same logical design densely and sparsely yields the same tree
/// structure and the same predictions.
#[test]
fn classifier_sparse_fit_equals_dense_fit() {
    let (dense, sparse, labels) = classifier_fixtures();
    let dense_model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(&dense, labels.view())
        .expect("dense fit succeeds");
    let sparse_model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(sparse.view(), labels.view())
        .expect("sparse fit succeeds");
    assert_eq!(dense_model.tree(), sparse_model.tree());
    assert_eq!(
        dense_model.feature_importances(),
        sparse_model.feature_importances()
    );
    assert_eq!(
        dense_model.predict_labels(dense.view()).expect("dense"),
        sparse_model.predict_labels(sparse.view()).expect("sparse")
    );
    assert_eq!(
        dense_model
            .predict_probabilities(dense.view())
            .expect("dense"),
        sparse_model
            .predict_probabilities(sparse.view())
            .expect("sparse")
    );
}

/// Regression parity holds for the tree and the predicted responses.
#[test]
fn regressor_sparse_fit_equals_dense_fit() {
    let dense = array![
        [0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let sparse = CsMat::new(
        (4, 3),
        vec![0, 1, 2, 3, 4],
        vec![1, 1, 0, 0],
        vec![2.0, 1.0, 3.0, 4.0],
    );
    let responses = array![1.0, 2.0, 3.0, 4.0];
    let dense_model = SKDecisionTreeRegressor::<f64>::build()
        .build()
        .expect("builds")
        .fit(&dense, responses.view())
        .expect("dense fit succeeds");
    let sparse_model = SKDecisionTreeRegressor::<f64>::build()
        .build()
        .expect("builds")
        .fit(sparse.view(), responses.view())
        .expect("sparse fit succeeds");
    assert_eq!(dense_model.tree(), sparse_model.tree());
    assert_eq!(
        dense_model.predict(dense.view()).expect("dense"),
        sparse_model.predict(sparse.view()).expect("sparse")
    );
}

/// A wide, mostly-zero sparse design fits from its stored entries without
/// materializing a dense copy.
#[test]
fn sparse_fit_handles_a_wide_sparse_design() {
    let rows = 2_000usize;
    let columns = 50usize;
    let indptr: Vec<usize> = (0..=rows).collect();
    let indices: Vec<usize> = (0..rows).map(|row| row % columns).collect();
    let data: Vec<f64> = (0..rows).map(|row| (row % 7) as f64).collect();
    let sparse = CsMat::new((rows, columns), indptr, indices, data);
    let labels: Vec<i64> = (0..rows).map(|row| (row % 2) as i64).collect();
    let labels = ndarray::Array1::from_vec(labels);
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(3)
        .build()
        .expect("builds")
        .fit(sparse.view(), labels.view())
        .expect("sparse fit succeeds");
    assert_eq!(model.feature_count(), columns);
    assert!(model.node_count() > 1);
}
