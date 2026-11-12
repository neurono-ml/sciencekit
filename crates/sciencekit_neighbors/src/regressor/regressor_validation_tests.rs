//! Companion validation tests for the neighbor regressor
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKRegressorPredictor, SKSupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKKNeighborsRegressor;
use crate::neighbors::{SKNeighborsAlgorithm, SKNeighborsMetric};

fn small_regressor() -> SKKNeighborsRegressor<f64> {
    SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
}

fn labelled(rows: usize) -> (Array2<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| row as f64 + column as f64);
    let responses = Array1::from_shape_fn(rows, |row| row as f64);
    (features, responses)
}

fn sparse_with_nan() -> CsMat<f64> {
    let mut triplet = TriMat::new((4, 3));
    triplet.add_triplet(0, 0, 1.0);
    triplet.add_triplet(1, 2, f64::NAN);
    triplet.add_triplet(2, 1, 2.0);
    triplet.to_csr()
}

#[test]
fn validation_regressor_rejects_empty_training_input() {
    let features = Array2::<f64>::zeros((0, 2));
    let responses = Array1::<f64>::zeros(0);
    let outcome = small_regressor().fit(features.view(), responses.view());

    match outcome {
        Err(SKError::EmptyInput { operation }) => assert_eq!(operation, "fit"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

#[test]
fn validation_regressor_rejects_a_neighbor_count_above_the_training_rows() {
    let (features, responses) = labelled(3);
    let outcome = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(4)
        .build()
        .unwrap()
        .fit(features.view(), responses.view());

    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "nearest_neighbors_count");
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn validation_regressor_rejects_a_feature_count_mismatch_at_predict() {
    let (features, responses) = labelled(6);
    let model = small_regressor()
        .fit(features.view(), responses.view())
        .unwrap();
    let queries = array![[0.0_f64, 0.0, 0.0]];

    match model.predict(queries.view()) {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(found, vec![1, 3]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn validation_regressor_rejects_non_finite_training_values() {
    let mut features = Array2::from_shape_fn((4, 2), |(row, column)| row as f64 + column as f64);
    features[[0, 1]] = f64::NAN;
    let responses = Array1::from_shape_fn(4, |row| row as f64);
    let outcome = small_regressor().fit(features.view(), responses.view());

    match outcome {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 1),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_regressor_rejects_non_finite_queries() {
    let (features, responses) = labelled(6);
    let model = small_regressor()
        .fit(features.view(), responses.view())
        .unwrap();
    let queries = array![[f64::INFINITY, 0.0]];

    match model.predict(queries.view()) {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 0),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_regressor_rejects_sparse_nan_input() {
    let responses = Array1::from_shape_fn(4, |row| row as f64);
    let outcome = small_regressor().fit(sparse_with_nan().view(), responses.view());

    match outcome {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 5),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_regressor_rejects_an_unsupported_metric_algorithm_pair() {
    let (features, responses) = labelled(6);
    let outcome = SKKNeighborsRegressor::<f64>::build()
        .algorithm(SKNeighborsAlgorithm::BallTree)
        .metric(SKNeighborsMetric::Cosine)
        .build()
        .unwrap()
        .fit(features.view(), responses.view());

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

#[test]
fn validation_regressor_rejects_empty_prediction_batches() {
    let (features, responses) = labelled(6);
    let model = small_regressor()
        .fit(features.view(), responses.view())
        .unwrap();
    let queries = Array2::<f64>::zeros((0, 2));

    match model.predict(queries.view()) {
        Err(SKError::EmptyInput { operation }) => assert_eq!(operation, "predict"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}
