//! Companion tests for the regressor builder defaults and validation
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::SKKNeighborsRegressor;
use crate::neighbors::{SKNeighborsAlgorithm, SKNeighborsMetric, SKNeighborsWeights};

#[test]
fn regressor_builder_defaults_match_the_documented_surface() {
    let estimator = SKKNeighborsRegressor::<f64>::build()
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
fn regressor_builder_records_every_configured_value() {
    let estimator = SKKNeighborsRegressor::<f32>::build()
        .nearest_neighbors_count(9)
        .weights(SKNeighborsWeights::Distance)
        .algorithm(SKNeighborsAlgorithm::KdTree)
        .metric(SKNeighborsMetric::Chebyshev)
        .leaf_size(12)
        .backend(SKBackendKind::NdArrayLinalg)
        .execution_mode(SKExecutionMode::InProcessAsynchronous)
        .build()
        .expect("the configuration is valid");

    assert_eq!(estimator.nearest_neighbors_count(), 9);
    assert_eq!(estimator.weights(), SKNeighborsWeights::Distance);
    assert_eq!(estimator.algorithm(), SKNeighborsAlgorithm::KdTree);
    assert_eq!(estimator.metric(), SKNeighborsMetric::Chebyshev);
    assert_eq!(estimator.leaf_size(), 12);
    assert_eq!(estimator.backend(), SKBackendKind::NdArrayLinalg);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessAsynchronous
    );
}

#[test]
fn regressor_builder_rejects_zero_neighbors() {
    let outcome = SKKNeighborsRegressor::<f64>::build()
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
fn regressor_builder_rejects_zero_leaf_size() {
    let outcome = SKKNeighborsRegressor::<f64>::build().leaf_size(0).build();

    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => assert_eq!(name, "leaf_size"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}
