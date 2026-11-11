//! Companion tests for `SKDBSCANBuilder` defaults, setters and validation
//! (spec `dbscan`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::{SKDBSCAN, SKDBSCANAlgorithm, SKDBSCANBuilder, SKDBSCANMetric};

#[test]
fn dbscan_builder_defaults_match_the_documented_configuration() {
    let estimator = SKDBSCAN::<f64>::build().build().unwrap();

    assert_eq!(estimator.eps(), 0.5);
    assert_eq!(estimator.minimum_samples(), 5);
    assert_eq!(estimator.metric(), SKDBSCANMetric::Euclidean);
    assert_eq!(estimator.algorithm(), SKDBSCANAlgorithm::Auto);
    assert_eq!(estimator.leaf_size(), 40);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn dbscan_builder_overrides_are_exposed_on_the_estimator() {
    let estimator = SKDBSCANBuilder::<f32>::default()
        .eps(1.25_f32)
        .minimum_samples(8)
        .metric(SKDBSCANMetric::Manhattan)
        .algorithm(SKDBSCANAlgorithm::BallTree)
        .leaf_size(16)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert_eq!(estimator.eps(), 1.25_f32);
    assert_eq!(estimator.minimum_samples(), 8);
    assert_eq!(estimator.metric(), SKDBSCANMetric::Manhattan);
    assert_eq!(estimator.algorithm(), SKDBSCANAlgorithm::BallTree);
    assert_eq!(estimator.leaf_size(), 16);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}

fn assert_invalid(error: SKError, expected_name: &'static str) {
    match error {
        SKError::InvalidHyperparameter { name, reason } => {
            assert_eq!(name, expected_name);
            assert!(!reason.is_empty());
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn dbscan_builder_rejects_invalid_hyperparameters_and_incompatible_pairs() {
    for bad_eps in [0.0_f64, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_invalid(
            SKDBSCAN::<f64>::build().eps(bad_eps).build().unwrap_err(),
            "eps",
        );
    }
    assert_invalid(
        SKDBSCAN::<f64>::build()
            .minimum_samples(0)
            .build()
            .unwrap_err(),
        "minimum_samples",
    );
    assert_invalid(
        SKDBSCAN::<f64>::build().leaf_size(0).build().unwrap_err(),
        "leaf_size",
    );

    for tree_algorithm in [SKDBSCANAlgorithm::KdTree, SKDBSCANAlgorithm::BallTree] {
        match SKDBSCAN::<f64>::build()
            .algorithm(tree_algorithm)
            .metric(SKDBSCANMetric::Cosine)
            .build()
            .unwrap_err()
        {
            SKError::UnsupportedRepresentation {
                representation,
                suggestion,
            } => {
                assert_eq!(representation, "cosine distance");
                assert!(suggestion.contains("BruteForce"));
            }
            other => panic!("expected UnsupportedRepresentation, found {other:?}"),
        }
    }
}

#[test]
fn dbscan_builder_builds_the_same_configuration_through_the_trait() {
    let mut builder = SKDBSCAN::<f64>::build();
    SKBuilder::execution_mode(&mut builder, SKExecutionMode::InProcessAsynchronous);
    let estimator = SKBuilder::build(builder).unwrap();
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessAsynchronous
    );
}
