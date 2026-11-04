//! Companion tests for the huber and insensitive regressor losses (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKRegressorPredictor, SKSupervisedFit};

use super::{SKSGDRegressor, SKSGDRegressorCoefficients};
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate, SKSGDRegressorLoss};
use crate::test_datasets::{
    assert_close, assert_close_scalar, outlier_regression_design, regression_design,
};

fn constant_rate(loss: SKSGDRegressorLoss, epsilon: f64) -> SKSGDRegressor<f64> {
    SKSGDRegressor::<f64>::build()
        .loss(loss)
        .epsilon(epsilon)
        .alpha(0.0)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .shuffle(false)
        .tolerance(1e-12)
        .maximum_number_of_iterations(200)
        .number_of_epochs_without_improvement(3)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
}

#[test]
fn a_large_epsilon_huber_matches_squared_error() {
    let (features, targets) = regression_design();
    let squared = constant_rate(SKSGDRegressorLoss::SquaredError, 0.1)
        .fit(features.view(), targets.view())
        .unwrap();
    let huber = constant_rate(SKSGDRegressorLoss::Huber, 1e6)
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close(
        huber.coefficients().as_slice().unwrap(),
        squared.coefficients().as_slice().unwrap(),
        1e-12,
    );
    assert_close_scalar(huber.intercept(), squared.intercept(), 1e-12);
}

#[test]
fn the_insensitive_band_keeps_the_start() {
    let (features, targets) = regression_design();
    let start = SKSGDRegressorCoefficients::new(Array1::from_vec(vec![1.5, -2.0]).view(), 3.0);
    for loss in [
        SKSGDRegressorLoss::EpsilonInsensitive,
        SKSGDRegressorLoss::SquaredEpsilonInsensitive,
    ] {
        let model = SKSGDRegressor::<f64>::build()
            .loss(loss)
            .epsilon(1e9)
            .alpha(0.0)
            .warm_start_from(start.clone())
            .shuffle(false)
            .maximum_number_of_iterations(5)
            .convergence_policy(SKConvergencePolicy::Permissive)
            .build()
            .unwrap()
            .fit(features.view(), targets.view())
            .unwrap();
        assert_close(
            model.coefficients().as_slice().unwrap(),
            &[1.5, -2.0],
            1e-12,
        );
        assert_close_scalar(model.intercept(), 3.0, 1e-12);
    }
}

#[test]
fn huber_resists_an_outlier_better_than_squared_error() {
    let (features, corrupted) = outlier_regression_design();
    let clean = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let fit = |loss: SKSGDRegressorLoss, epsilon: f64, targets: &Array1<f64>| {
        SKSGDRegressor::<f64>::build()
            .loss(loss)
            .epsilon(epsilon)
            .alpha(0.0)
            .learning_rate(SKSGDLearningRate::Constant)
            .initial_learning_rate(0.01)
            .shuffle(false)
            .tolerance(1e-14)
            .maximum_number_of_iterations(4_000)
            .number_of_epochs_without_improvement(20)
            .convergence_policy(SKConvergencePolicy::Permissive)
            .build()
            .unwrap()
            .fit(features.view(), targets.view())
            .unwrap()
    };
    let reference = fit(SKSGDRegressorLoss::SquaredError, 0.1, &clean);
    let squared = fit(SKSGDRegressorLoss::SquaredError, 0.1, &corrupted);
    let huber = fit(SKSGDRegressorLoss::Huber, 1.0, &corrupted);
    let distance = |model: &super::SKSGDRegressorModel<f64>| {
        (model.coefficients()[0] - reference.coefficients()[0]).abs()
    };
    assert!(
        distance(&huber) < distance(&squared),
        "huber moved {} from the clean fit, squared error moved {}",
        distance(&huber),
        distance(&squared)
    );
}

#[test]
fn epsilon_zero_insensitive_equals_the_absolute_error_step() {
    let features = array![[1.0]];
    let targets = array![2.0];
    let learning_rate = 0.1;
    let model = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::EpsilonInsensitive)
        .epsilon(0.0)
        .alpha(0.0)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(learning_rate)
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    // The absolute-error slope is sign(0 − 2) = −1, so w = η and b = η.
    assert_close(
        model.coefficients().as_slice().unwrap(),
        &[learning_rate],
        1e-12,
    );
    assert_close_scalar(model.intercept(), learning_rate, 1e-12);
}

#[test]
fn zero_epsilon_squared_insensitive_matches_squared_error() {
    let (features, targets) = regression_design();
    let squared = constant_rate(SKSGDRegressorLoss::SquaredError, 0.1)
        .fit(features.view(), targets.view())
        .unwrap();
    let insensitive = constant_rate(SKSGDRegressorLoss::SquaredEpsilonInsensitive, 0.0)
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close(
        insensitive.coefficients().as_slice().unwrap(),
        squared.coefficients().as_slice().unwrap(),
        1e-12,
    );
    assert_close_scalar(insensitive.intercept(), squared.intercept(), 1e-12);
    let predicted = insensitive.predict(features.view()).unwrap();
    assert!(predicted.iter().all(|value| value.is_finite()));
}
