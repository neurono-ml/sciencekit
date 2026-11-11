//! Companion tests for `SKTruncatedSVDBuilder` and `SKTruncatedSVD` configuration.

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionContext, SKExecutionMode};

use super::{SKTruncatedSVD, SKTruncatedSVDAlgorithm, SKTruncatedSVDBuilder};

#[test]
fn truncated_builder_defaults_match_the_documented_configuration() {
    let estimator = SKTruncatedSVD::<f64>::build().build().unwrap();

    assert_eq!(estimator.number_of_components(), 2);
    assert_eq!(estimator.algorithm(), SKTruncatedSVDAlgorithm::Randomized);
    assert_eq!(estimator.number_of_iterations(), 5);
    assert_eq!(estimator.tolerance(), 0.0);
    assert_eq!(estimator.random_seed(), 0);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);

    let plan = estimator
        .resolve_plan(&SKExecutionContext::real())
        .expect("default plan resolves");
    assert!(plan.parallelism >= 1);
}

#[test]
fn truncated_builder_overrides_are_exposed_on_the_estimator_and_via_trait() {
    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(6)
        .algorithm(SKTruncatedSVDAlgorithm::Randomized)
        .number_of_iterations(9)
        .tolerance(1e-4)
        .random_seed(42)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert_eq!(estimator.number_of_components(), 6);
    assert_eq!(estimator.algorithm(), SKTruncatedSVDAlgorithm::Randomized);
    assert_eq!(estimator.number_of_iterations(), 9);
    assert_eq!(estimator.tolerance(), 1e-4);
    assert_eq!(estimator.random_seed(), 42);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let mut via_trait = SKTruncatedSVDBuilder::<f32>::default();
    SKBuilder::execution_mode(&mut via_trait, SKExecutionMode::InProcessAsynchronous);
    let built = SKBuilder::build(via_trait).unwrap();
    assert_eq!(built.number_of_components(), 2);
    assert_eq!(
        built.execution_mode(),
        SKExecutionMode::InProcessAsynchronous
    );
}

fn assert_invalid_hyperparameter(error: SKError, expected_name: &'static str) {
    match error {
        SKError::InvalidHyperparameter { name, reason } => {
            assert_eq!(name, expected_name);
            assert!(!reason.is_empty());
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn truncated_builder_rejects_invalid_hyperparameters() {
    assert_invalid_hyperparameter(
        SKTruncatedSVD::<f64>::build()
            .number_of_components(0)
            .build()
            .unwrap_err(),
        "number_of_components",
    );
    assert_invalid_hyperparameter(
        SKTruncatedSVD::<f64>::build()
            .algorithm(SKTruncatedSVDAlgorithm::Arpack)
            .build()
            .unwrap_err(),
        "algorithm",
    );
    assert_invalid_hyperparameter(
        SKTruncatedSVD::<f64>::build()
            .number_of_iterations(0)
            .build()
            .unwrap_err(),
        "number_of_iterations",
    );
    assert_invalid_hyperparameter(
        SKTruncatedSVD::<f64>::build()
            .tolerance(-1e-6)
            .build()
            .unwrap_err(),
        "tolerance",
    );
    assert_invalid_hyperparameter(
        SKTruncatedSVD::<f64>::build()
            .tolerance(f64::NAN)
            .build()
            .unwrap_err(),
        "tolerance",
    );
    assert_invalid_hyperparameter(
        SKTruncatedSVD::<f64>::build()
            .tolerance(f64::INFINITY)
            .build()
            .unwrap_err(),
        "tolerance",
    );
}
