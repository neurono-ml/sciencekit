//! Companion tests for classifier fit: stored design and resolved strategy
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKSupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKKNeighborsClassifier;
use crate::neighbors::{SKNeighborsAlgorithm, SKSearchStrategy};

fn dense_design(rows: usize) -> (Array2<f64>, Array1<i64>) {
    let features = Array2::from_shape_fn((rows, 3), |(row, column)| {
        ((row * 7 + column * 13) % 23) as f64 / 3.0
    });
    let labels = Array1::from_shape_fn(rows, |row| (row % 2) as i64);
    (features, labels)
}

fn sparse_design(rows: usize) -> CsMat<f64> {
    let mut triplet = TriMat::new((rows, 3));
    for row in 0..rows {
        triplet.add_triplet(row, row % 3, 1.0 + row as f64);
        triplet.add_triplet(row, (row + 1) % 3, 0.5);
    }
    triplet.to_csr()
}

#[test]
fn classifier_fit_dense_reports_the_auto_tree_strategy() {
    let (features, labels) = dense_design(100);
    let model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.search_strategy(), SKSearchStrategy::KdTree);
    assert_eq!(model.training_rows(), 100);
    assert_eq!(model.number_of_features(), 3);
    assert_eq!(model.number_of_classes(), 2);
    assert_eq!(model.nearest_neighbors_count(), 5);
}

#[test]
fn classifier_fit_sparse_reports_the_brute_force_strategy() {
    let features = sparse_design(30);
    let labels = Array1::from_shape_fn(30, |row| (row % 2) as i64);
    let model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.search_strategy(), SKSearchStrategy::BruteForce);
    assert_eq!(model.training_rows(), 30);
    assert_eq!(model.number_of_features(), 3);
}

#[test]
fn classifier_fit_small_dense_data_uses_brute_force() {
    let (features, labels) = dense_design(20);
    let model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.search_strategy(), SKSearchStrategy::BruteForce);
}

#[test]
fn classifier_fit_carries_the_backend_and_execution_mode_into_the_model() {
    let (features, labels) = dense_design(20);
    let model = SKKNeighborsClassifier::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(model.execution_mode(), SKExecutionMode::InProcessSynchronous);
}

#[test]
fn classifier_fit_rejects_an_explicit_tree_over_sparse_data() {
    let features = sparse_design(30);
    let labels = Array1::from_shape_fn(30, |row| (row % 2) as i64);
    let outcome = SKKNeighborsClassifier::<f64>::build()
        .algorithm(SKNeighborsAlgorithm::KdTree)
        .build()
        .unwrap()
        .fit(features.view(), labels.view());

    match outcome {
        Err(SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        }) => {
            assert_eq!(representation, "csr");
            assert!(suggestion.contains("BruteForce"));
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}

#[test]
fn classifier_fit_rejects_a_target_length_mismatch() {
    let (features, _) = dense_design(20);
    let labels = Array1::from_shape_fn(10, |row| (row % 2) as i64);
    let outcome = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view());

    match outcome {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, vec![20, 1]);
            assert_eq!(found, vec![10, 1]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn classifier_fit_rejects_an_excessive_neighbor_count() {
    let (features, labels) = dense_design(4);
    let outcome = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(5)
        .build()
        .unwrap()
        .fit(features.view(), labels.view());

    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "nearest_neighbors_count");
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn classifier_fit_rejects_continuous_targets() {
    let (features, _) = dense_design(20);
    let continuous = Array1::from_elem(20, 0.5_f64);
    let outcome = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), continuous.view());

    match outcome {
        Err(SKError::UnsupportedRepresentation { representation, .. }) => {
            assert_eq!(representation, "continuous");
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}
