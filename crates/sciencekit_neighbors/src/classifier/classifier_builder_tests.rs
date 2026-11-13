//! Companion tests for the classifier builder defaults and validation
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::SKKNeighborsClassifier;
use crate::neighbors::{SKNeighborsAlgorithm, SKNeighborsMetric, SKNeighborsWeights};

#[test]
fn classifier_builder_defaults_match_the_documented_surface() {
    let estimator = SKKNeighborsClassifier::<f64>::build()
        .build()
        .expect("the defaults are valid");

    assert_eq!(estimator.nearest_neighbors_count(), 5);
    assert_eq!(estimator.weights(), SKNeighborsWeights::Uniform);
    assert_eq!(estimator.algorithm(), SKNeighborsAlgorithm::Auto);
    assert_eq!(estimator.metric(), SKNeighborsMetric::Euclidean);
    assert_eq!(estimator.leaf_size(), 40);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn classifier_builder_records_every_configured_value() {
    let estimator = SKKNeighborsClassifier::<f32>::build()
        .nearest_neighbors_count(11)
        .weights(SKNeighborsWeights::Distance)
        .algorithm(SKNeighborsAlgorithm::BallTree)
        .metric(SKNeighborsMetric::Manhattan)
        .leaf_size(7)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("the configuration is valid");

    assert_eq!(estimator.nearest_neighbors_count(), 11);
    assert_eq!(estimator.weights(), SKNeighborsWeights::Distance);
    assert_eq!(estimator.algorithm(), SKNeighborsAlgorithm::BallTree);
    assert_eq!(estimator.metric(), SKNeighborsMetric::Manhattan);
    assert_eq!(estimator.leaf_size(), 7);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn classifier_builder_rejects_zero_neighbors() {
    let outcome = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(0)
        .build();

    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "nearest_neighbors_count");
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn classifier_builder_rejects_zero_leaf_size() {
    let outcome = SKKNeighborsClassifier::<f64>::build().leaf_size(0).build();

    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => assert_eq!(name, "leaf_size"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}
