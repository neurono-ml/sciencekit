//! Companion tests for the SGD regressor builder (spec
//! `stochastic-gradient-regressor`).

use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKRegressorPredictor, SKSupervisedFit,
};

use super::SKSGDRegressor;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate, SKSGDPenalty, SKSGDRegressorLoss};
use crate::test_datasets::regression_design;

#[test]
fn builder_defaults_are_documented() {
    let estimator = SKSGDRegressor::<f64>::build().build().unwrap();
    assert_eq!(estimator.loss(), SKSGDRegressorLoss::SquaredError);
    assert_eq!(estimator.penalty(), SKSGDPenalty::L2);
    assert_eq!(estimator.alpha(), 1e-4);
    assert!(estimator.include_intercept());
    assert_eq!(estimator.learning_rate(), SKSGDLearningRate::InverseScaling);
    assert_eq!(estimator.initial_learning_rate(), 0.01);
    assert_eq!(estimator.epsilon(), 0.1);
    assert!(!estimator.early_stopping());
    assert_eq!(estimator.convergence_policy(), SKConvergencePolicy::Strict);
    assert!(estimator.shuffle());
    assert_eq!(estimator.random_seed(), 0);
    assert!(!estimator.average());
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
}

#[test]
fn every_knob_is_settable() {
    let estimator = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::Huber)
        .penalty(SKSGDPenalty::ElasticNet)
        .alpha(0.5)
        .l1_ratio(0.25)
        .include_intercept(false)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.2)
        .inverse_scaling_power(0.75)
        .tolerance(1e-5)
        .maximum_number_of_iterations(42)
        .number_of_epochs_without_improvement(3)
        .early_stopping(true)
        .validation_fraction(0.2)
        .shuffle(false)
        .random_seed(7)
        .average(true)
        .epsilon(0.75)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap();
    assert_eq!(estimator.loss(), SKSGDRegressorLoss::Huber);
    assert_eq!(estimator.penalty(), SKSGDPenalty::ElasticNet);
    assert_eq!(estimator.alpha(), 0.5);
    assert_eq!(estimator.l1_ratio(), 0.25);
    assert!(!estimator.include_intercept());
    assert_eq!(estimator.learning_rate(), SKSGDLearningRate::Constant);
    assert_eq!(estimator.initial_learning_rate(), 0.2);
    assert_eq!(estimator.inverse_scaling_power(), 0.75);
    assert_eq!(estimator.tolerance(), 1e-5);
    assert_eq!(estimator.maximum_number_of_iterations(), 42);
    assert_eq!(estimator.number_of_epochs_without_improvement(), 3);
    assert!(estimator.early_stopping());
    assert_eq!(estimator.validation_fraction(), 0.2);
    assert!(!estimator.shuffle());
    assert_eq!(estimator.random_seed(), 7);
    assert!(estimator.average());
    assert_eq!(estimator.epsilon(), 0.75);
    assert_eq!(
        estimator.convergence_policy(),
        SKConvergencePolicy::Permissive
    );
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
}

fn invalid_name(result: Result<SKSGDRegressor<f64>, SKError>) -> &'static str {
    match result.unwrap_err() {
        SKError::InvalidHyperparameter { name, .. } => name,
        other => panic!("expected an invalid hyperparameter, got {other:?}"),
    }
}

#[test]
fn invalid_configurations_are_rejected() {
    assert_eq!(
        invalid_name(SKSGDRegressor::<f64>::build().alpha(-1.0).build()),
        "alpha"
    );
    assert_eq!(
        invalid_name(SKSGDRegressor::<f64>::build().l1_ratio(2.0).build()),
        "l1_ratio"
    );
    assert_eq!(
        invalid_name(SKSGDRegressor::<f64>::build().tolerance(0.0).build()),
        "tolerance"
    );
    assert_eq!(
        invalid_name(
            SKSGDRegressor::<f64>::build()
                .maximum_number_of_iterations(0)
                .build()
        ),
        "maximum_number_of_iterations"
    );
    assert_eq!(
        invalid_name(
            SKSGDRegressor::<f64>::build()
                .number_of_epochs_without_improvement(0)
                .build()
        ),
        "number_of_epochs_without_improvement"
    );
    assert_eq!(
        invalid_name(
            SKSGDRegressor::<f64>::build()
                .validation_fraction(1.0)
                .build()
        ),
        "validation_fraction"
    );
    assert_eq!(
        invalid_name(
            SKSGDRegressor::<f64>::build()
                .early_stopping(true)
                .maximum_number_of_iterations(1)
                .build()
        ),
        "early_stopping"
    );
    assert_eq!(
        invalid_name(
            SKSGDRegressor::<f64>::build()
                .learning_rate(SKSGDLearningRate::Optimal)
                .alpha(0.0)
                .build()
        ),
        "learning_rate"
    );
    assert_eq!(
        invalid_name(SKSGDRegressor::<f64>::build().epsilon(-1.0).build()),
        "epsilon"
    );
}

#[test]
fn configuration_flows_into_the_model() {
    let (features, targets) = regression_design();
    let model = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::Huber)
        .epsilon(2.0)
        .include_intercept(false)
        .average(true)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.loss(), SKSGDRegressorLoss::Huber);
    assert_eq!(model.epsilon(), 2.0);
    assert!(!model.include_intercept());
    assert!(model.average());
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert!(model.epochs_taken() > 0);
}

#[test]
fn the_same_seed_reproduces_and_the_model_serves_threads() {
    let (features, targets) = regression_design();
    let first = SKSGDRegressor::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let second = SKSGDRegressor::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(first.coefficients(), second.coefficients());
    let expected = first.predict(features.view()).unwrap();
    std::thread::scope(|scope| {
        let model = &first;
        let features = features.view();
        let expected = &expected;
        for _ in 0..4 {
            scope.spawn(move || {
                let predicted = model.predict(features).unwrap();
                assert_eq!(&predicted, expected);
            });
        }
    });
}
