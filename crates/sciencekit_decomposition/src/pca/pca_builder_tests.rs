//! Companion tests for `SKPCABuilder` and `SKPCA` configuration.

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::{SKPCA, SKPCABuilder, SKPCASolver};

#[test]
fn pca_builder_defaults_match_the_documented_configuration() {
    let estimator = SKPCA::<f64>::build().build().unwrap();

    assert_eq!(estimator.number_of_components(), None);
    assert_eq!(estimator.svd_solver(), SKPCASolver::Auto);
    assert!(!estimator.whiten());
    assert_eq!(estimator.tolerance(), 0.0);
    assert_eq!(estimator.number_of_power_iterations(), 5);
    assert!(!estimator.allow_densification());
    assert_eq!(estimator.random_seed(), 0);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn pca_builder_overrides_are_exposed_on_the_estimator_and_via_trait() {
    let estimator = SKPCA::<f64>::build()
        .number_of_components(4)
        .svd_solver(SKPCASolver::Randomized)
        .whiten(true)
        .tolerance(1e-5)
        .number_of_power_iterations(7)
        .allow_densification(true)
        .random_seed(123)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert_eq!(estimator.number_of_components(), Some(4));
    assert_eq!(estimator.svd_solver(), SKPCASolver::Randomized);
    assert!(estimator.whiten());
    assert_eq!(estimator.tolerance(), 1e-5);
    assert_eq!(estimator.number_of_power_iterations(), 7);
    assert!(estimator.allow_densification());
    assert_eq!(estimator.random_seed(), 123);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let mut via_trait = SKPCABuilder::<f32>::default();
    SKBuilder::execution_mode(&mut via_trait, SKExecutionMode::InProcessAsynchronous);
    let built = SKBuilder::build(via_trait).unwrap();
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
fn pca_builder_rejects_invalid_hyperparameters() {
    assert_invalid_hyperparameter(
        SKPCA::<f64>::build()
            .number_of_components(0)
            .build()
            .unwrap_err(),
        "number_of_components",
    );
    assert_invalid_hyperparameter(
        SKPCA::<f64>::build().tolerance(-0.01).build().unwrap_err(),
        "tolerance",
    );
    assert_invalid_hyperparameter(
        SKPCA::<f64>::build()
            .tolerance(f64::NAN)
            .build()
            .unwrap_err(),
        "tolerance",
    );
    assert_invalid_hyperparameter(
        SKPCA::<f64>::build()
            .tolerance(f64::INFINITY)
            .build()
            .unwrap_err(),
        "tolerance",
    );
    assert_invalid_hyperparameter(
        SKPCA::<f64>::build()
            .number_of_power_iterations(0)
            .build()
            .unwrap_err(),
        "number_of_power_iterations",
    );
}
