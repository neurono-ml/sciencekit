//! Companion tests for the KMeans builder.

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::{SKKMeans, SKKMeansAlgorithm, SKKMeansInitialization};

#[test]
fn kmeans_builder_defaults_match_the_documented_configuration() {
    let estimator = SKKMeans::<f64>::build().build().unwrap();

    assert_eq!(estimator.number_of_clusters(), 8);
    assert_eq!(
        estimator.initialization(),
        SKKMeansInitialization::KMeansPlusPlus
    );
    assert_eq!(estimator.number_of_initializations(), 1);
    assert_eq!(estimator.maximum_number_of_iterations(), 300);
    assert_eq!(estimator.tolerance(), 1e-4);
    assert_eq!(estimator.algorithm(), SKKMeansAlgorithm::Auto);
    assert_eq!(estimator.random_seed(), 0);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn kmeans_builder_random_initialization_defaults_to_ten_initializations() {
    let estimator = SKKMeans::<f64>::build()
        .initialization(SKKMeansInitialization::Random)
        .build()
        .unwrap();

    assert_eq!(estimator.number_of_initializations(), 10);
}

#[test]
fn kmeans_builder_overrides_are_exposed_on_the_estimator() {
    let estimator = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(4)
        .maximum_number_of_iterations(50)
        .tolerance(1e-6)
        .algorithm(SKKMeansAlgorithm::Lloyd)
        .random_seed(99)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert_eq!(estimator.number_of_clusters(), 3);
    assert_eq!(estimator.initialization(), SKKMeansInitialization::Random);
    assert_eq!(estimator.number_of_initializations(), 4);
    assert_eq!(estimator.maximum_number_of_iterations(), 50);
    assert_eq!(estimator.tolerance(), 1e-6);
    assert_eq!(estimator.algorithm(), SKKMeansAlgorithm::Lloyd);
    assert_eq!(estimator.random_seed(), 99);
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
fn kmeans_builder_rejects_invalid_hyperparameters() {
    assert_invalid(
        SKKMeans::<f64>::build()
            .number_of_clusters(0)
            .build()
            .unwrap_err(),
        "number_of_clusters",
    );
    assert_invalid(
        SKKMeans::<f64>::build()
            .maximum_number_of_iterations(0)
            .build()
            .unwrap_err(),
        "maximum_number_of_iterations",
    );
    assert_invalid(
        SKKMeans::<f64>::build()
            .tolerance(-1.0)
            .build()
            .unwrap_err(),
        "tolerance",
    );
    assert_invalid(
        SKKMeans::<f64>::build()
            .tolerance(f64::NAN)
            .build()
            .unwrap_err(),
        "tolerance",
    );
    assert_invalid(
        SKKMeans::<f64>::build()
            .number_of_initializations(0)
            .build()
            .unwrap_err(),
        "number_of_initializations",
    );
}

#[test]
fn kmeans_builder_builds_the_same_configuration_through_the_trait() {
    let mut builder = SKKMeans::<f64>::build();
    SKBuilder::execution_mode(&mut builder, SKExecutionMode::InProcessAsynchronous);
    let estimator = SKBuilder::build(builder).unwrap();
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessAsynchronous
    );
}
