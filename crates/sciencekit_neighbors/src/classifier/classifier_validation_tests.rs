//! Companion validation tests for the neighbor classifier
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKError, SKSupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKKNeighborsClassifier;
use crate::neighbors::{SKNeighborsAlgorithm, SKNeighborsMetric};

fn small_classifier() -> SKKNeighborsClassifier<f64> {
    SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
}

fn labelled(rows: usize) -> (Array2<f64>, Array1<i64>) {
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| row as f64 + column as f64);
    let labels = Array1::from_shape_fn(rows, |row| (row % 2) as i64);
    (features, labels)
}

#[test]
fn validation_rejects_empty_training_input() {
    let features = Array2::<f64>::zeros((0, 2));
    let labels = Array1::<i64>::zeros(0);
    let outcome = small_classifier().fit(features.view(), labels.view());

    match outcome {
        Err(SKError::EmptyInput { operation }) => assert_eq!(operation, "fit"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

#[test]
fn validation_rejects_zero_feature_columns() {
    let features = Array2::<f64>::zeros((3, 0));
    let labels = Array1::<i64>::zeros(3);
    let outcome = small_classifier().fit(features.view(), labels.view());

    match outcome {
        Err(SKError::EmptyInput { operation }) => assert_eq!(operation, "fit"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

#[test]
fn validation_rejects_zero_neighbors_at_build() {
    match SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(0)
        .build()
    {
        Err(SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "nearest_neighbors_count");
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn validation_rejects_a_neighbor_count_above_the_training_rows() {
    let (features, labels) = labelled(3);
    let outcome = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(4)
        .build()
        .unwrap()
        .fit(features.view(), labels.view());

    match outcome {
        Err(SKError::InvalidHyperparameter { name, reason }) => {
            assert_eq!(name, "nearest_neighbors_count");
            assert!(reason.contains("4") && reason.contains("3"));
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn validation_rejects_a_feature_count_mismatch_at_predict() {
    let (features, labels) = labelled(6);
    let model = small_classifier()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[0.0_f64, 0.0, 0.0]];

    match model.predict_labels(queries.view()) {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(found, vec![1, 3]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn validation_rejects_non_finite_training_values() {
    let mut features = Array2::from_shape_fn((4, 2), |(row, column)| row as f64 + column as f64);
    features[[1, 1]] = f64::INFINITY;
    let labels = Array1::from_shape_fn(4, |row| (row % 2) as i64);
    let outcome = small_classifier().fit(features.view(), labels.view());

    match outcome {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 3),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_rejects_non_finite_query_values() {
    let (features, labels) = labelled(6);
    let model = small_classifier()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[1.0_f64, f64::NAN]];

    match model.predict_labels(queries.view()) {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 1),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

fn sparse_with_nan() -> CsMat<f64> {
    let mut triplet = TriMat::new((4, 3));
    triplet.add_triplet(0, 0, 1.0);
    triplet.add_triplet(1, 2, f64::NAN);
    triplet.add_triplet(2, 1, 2.0);
    triplet.to_csr()
}

#[test]
fn validation_rejects_sparse_nan_training_values() {
    let labels = Array1::from_shape_fn(4, |row| (row % 2) as i64);
    let outcome = small_classifier().fit(sparse_with_nan().view(), labels.view());

    match outcome {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 5),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_rejects_sparse_nan_queries() {
    let features = Array2::from_shape_fn((4, 3), |(row, column)| row as f64 + column as f64);
    let labels = Array1::from_shape_fn(4, |row| (row % 2) as i64);
    let model = small_classifier()
        .fit(features.view(), labels.view())
        .unwrap();

    match model.predict_labels(sparse_with_nan().view()) {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 5),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_rejects_empty_prediction_batches() {
    let (features, labels) = labelled(6);
    let model = small_classifier()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = Array2::<f64>::zeros((0, 2));

    match model.predict_labels(queries.view()) {
        Err(SKError::EmptyInput { operation }) => assert_eq!(operation, "predict"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

#[test]
fn validation_rejects_an_unsupported_metric_algorithm_pair() {
    let (features, labels) = labelled(6);
    let outcome = SKKNeighborsClassifier::<f64>::build()
        .algorithm(SKNeighborsAlgorithm::KdTree)
        .metric(SKNeighborsMetric::Cosine)
        .build()
        .unwrap()
        .fit(features.view(), labels.view());

    match outcome {
        Err(SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        }) => {
            assert_eq!(representation, "cosine distance");
            assert!(suggestion.contains("BruteForce"));
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}
