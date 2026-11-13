//! Companion tests for the nearest-neighbor imputer builder
//! (spec `nearest-neighbor-imputation`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use crate::knn::core_implementation::SKKNNImputer;
use crate::knn::weights::SKImputationWeights;

#[test]
fn knn_builder_defaults_are_the_documented_ones() {
    let imputer: SKKNNImputer<f64> = SKKNNImputer::build().build().unwrap();
    assert!(imputer.missing_values().is_nan());
    assert_eq!(imputer.nearest_neighbors_count(), 5);
    assert_eq!(imputer.weights(), SKImputationWeights::Uniform);
    assert!(!imputer.add_indicator());
    assert!(!imputer.keep_empty_features());
    assert_eq!(imputer.backend(), SKBackendKind::Faer);
    assert_eq!(imputer.execution_intent(), SKExecutionMode::Automatic);
}

#[test]
fn knn_builder_setters_reach_the_estimator() {
    let imputer: SKKNNImputer<f64> = SKKNNImputer::build()
        .missing_values(-999.0)
        .nearest_neighbors_count(3)
        .weights(SKImputationWeights::Distance)
        .add_indicator(true)
        .keep_empty_features(true)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(imputer.missing_values(), -999.0);
    assert_eq!(imputer.nearest_neighbors_count(), 3);
    assert_eq!(imputer.weights(), SKImputationWeights::Distance);
    assert!(imputer.add_indicator());
    assert!(imputer.keep_empty_features());
    assert_eq!(imputer.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        imputer.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn knn_builder_generic_over_the_scalar() {
    let imputer: SKKNNImputer<f32> = SKKNNImputer::build().build().unwrap();
    assert!(imputer.missing_values().is_nan());
}

#[test]
fn knn_builder_rejects_a_zero_neighbor_count() {
    let error = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(0)
        .build()
        .err()
        .expect("zero donors must be rejected");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "nearest_neighbors_count",
            ..
        }
    ));
}

#[test]
fn knn_builder_rejects_an_infinite_marker() {
    let error = SKKNNImputer::<f64>::build()
        .missing_values(f64::INFINITY)
        .build()
        .err()
        .expect("an infinite marker must be rejected");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "missing_values",
            ..
        }
    ));
}
