//! Companion tests for fit/assignment validation (spec `k-means`, decision 9).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKClustererPredictor, SKError, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKKMeans;

fn estimator(number_of_clusters: usize) -> SKKMeans<f64> {
    SKKMeans::<f64>::build()
        .number_of_clusters(number_of_clusters)
        .build()
        .unwrap()
}

fn csr(values: &[(usize, usize, f64)], shape: (usize, usize)) -> CsMat<f64> {
    let mut triplet = TriMat::new(shape);
    for (row, column, value) in values {
        triplet.add_triplet(*row, *column, *value);
    }
    triplet.to_csr()
}

#[test]
fn kmeans_validation_rejects_empty_input() {
    for data in [Array2::<f64>::zeros((0, 2)), Array2::<f64>::zeros((3, 0))] {
        match estimator(1).fit(&data).unwrap_err() {
            SKError::EmptyInput { operation } => assert_eq!(operation, "fit"),
            other => panic!("expected EmptyInput, found {other:?}"),
        }
    }
    let empty = csr(&[], (0, 2));
    match estimator(1).fit(empty.view()).unwrap_err() {
        SKError::EmptyInput { operation } => assert_eq!(operation, "fit"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

#[test]
fn kmeans_validation_rejects_more_clusters_than_samples() {
    let data = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0]];
    match estimator(4).fit(&data).unwrap_err() {
        SKError::InvalidHyperparameter { name, .. } => {
            assert_eq!(name, "number_of_clusters");
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn kmeans_validation_rejects_non_finite_training_values_at_their_position() {
    let not_a_number = array![[0.0_f64, 1.0], [2.0, f64::NAN]];
    let infinite = array![[0.0_f64, 1.0], [f64::INFINITY, 2.0]];

    for (data, expected) in [(not_a_number, 3_usize), (infinite, 2)] {
        match estimator(1).fit(&data).unwrap_err() {
            SKError::NonFiniteValue { position } => assert_eq!(position, expected),
            other => panic!("expected NonFiniteValue, found {other:?}"),
        }
    }
}

#[test]
fn kmeans_validation_rejects_non_finite_sparse_values_at_their_flat_position() {
    let sparse = csr(&[(0, 0, 1.0), (1, 2, f64::NAN)], (3, 3));
    match estimator(1).fit(sparse.view()).unwrap_err() {
        SKError::NonFiniteValue { position } => assert_eq!(position, 5),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn kmeans_validation_rejects_non_finite_assignment_values() {
    let fitted = estimator(2)
        .fit(&array![[0.0_f64], [10.0], [10.1]])
        .unwrap();
    let queries = array![[0.0_f64], [f64::INFINITY]];

    match fitted.predict_clusters(&queries).unwrap_err() {
        SKError::NonFiniteValue { position } => assert_eq!(position, 1),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
    match fitted.transform(&queries).unwrap_err() {
        SKError::NonFiniteValue { position } => assert_eq!(position, 1),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn kmeans_validation_rejects_invalid_builder_parameters() {
    for error in [
        SKKMeans::<f64>::build()
            .number_of_clusters(0)
            .build()
            .unwrap_err(),
        SKKMeans::<f64>::build()
            .maximum_number_of_iterations(0)
            .build()
            .unwrap_err(),
        SKKMeans::<f64>::build()
            .tolerance(-0.5)
            .build()
            .unwrap_err(),
    ] {
        match error {
            SKError::InvalidHyperparameter { name, .. } => assert!(!name.is_empty()),
            other => panic!("expected InvalidHyperparameter, found {other:?}"),
        }
    }
}
