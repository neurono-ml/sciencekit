//! Companion tests for the penalty portfolio and the OLS limit of the SGD
//! regressor (spec `stochastic-gradient-regressor`).

use ndarray::{Array1, Array2};
use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::{SKSGDRegressor, SKSGDRegressorCoefficients};
use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate, SKSGDPenalty, SKSGDRegressorLoss};
use crate::test_datasets::{assert_close, assert_close_scalar, regression_design};

fn fit_penalized(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    penalty: SKSGDPenalty,
    alpha: f64,
    l1_ratio: f64,
    warm_start: Option<SKSGDRegressorCoefficients<f64>>,
) -> super::SKSGDRegressorModel<f64> {
    let mut builder = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::SquaredError)
        .penalty(penalty)
        .alpha(alpha)
        .l1_ratio(l1_ratio)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .shuffle(false)
        .tolerance(1e-12)
        .maximum_number_of_iterations(8_000)
        .number_of_epochs_without_improvement(10)
        .convergence_policy(SKConvergencePolicy::Permissive);
    if let Some(source) = warm_start {
        builder = builder.warm_start_from(source);
    }
    builder
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
}

fn norm(model: &super::SKSGDRegressorModel<f64>) -> f64 {
    model.coefficients().dot(&model.coefficients()).sqrt()
}

#[test]
fn squared_error_converges_to_the_ols_solution() {
    let (features, targets) = regression_design();
    let reference = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let model = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::SquaredError)
        .alpha(0.0)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .shuffle(false)
        .tolerance(1e-12)
        .maximum_number_of_iterations(20_000)
        .number_of_epochs_without_improvement(3)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(model.converged());
    assert_close(
        model.coefficients().as_slice().unwrap(),
        reference.coefficients().as_slice().unwrap(),
        1e-4,
    );
    assert_close_scalar(model.intercept(), reference.intercept(), 1e-4);
}

#[test]
fn growing_l2_shrinks_the_coefficients_monotonically() {
    let (features, targets) = regression_design();
    let alphas = [0.0, 0.1, 1.0];
    let norms: Vec<f64> = alphas
        .iter()
        .map(|&alpha| {
            norm(&fit_penalized(
                &features,
                &targets,
                SKSGDPenalty::L2,
                alpha,
                0.15,
                None,
            ))
        })
        .collect();
    assert!(
        norms[0] > norms[1] && norms[1] > norms[2],
        "the L2 norms did not shrink monotonically: {norms:?}"
    );
}

#[test]
fn l1_produces_exact_zeros_on_an_irrelevant_feature() {
    let features = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0],
    )
    .unwrap();
    let targets = Array1::from_vec(vec![3.0, 6.0, 9.0, 12.0, 15.0]);
    let start = SKSGDRegressorCoefficients::new(Array1::from_vec(vec![0.0, 5.0]).view(), 0.0);
    let model = SKSGDRegressor::<f64>::build()
        .penalty(SKSGDPenalty::L1)
        .alpha(1.0)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.1)
        .shuffle(false)
        .maximum_number_of_iterations(500)
        .number_of_epochs_without_improvement(500)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .warm_start_from(start)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.coefficients()[1], 0.0);
    assert!(model.coefficients()[0].is_finite());
}

#[test]
fn elastic_net_limits_match_the_pure_penalties() {
    let features = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0],
    )
    .unwrap();
    let targets = Array1::from_vec(vec![3.0, 6.0, 9.0, 12.0, 15.0]);
    let start = SKSGDRegressorCoefficients::new(Array1::from_vec(vec![0.0, 5.0]).view(), 0.0);
    let l1 = fit_penalized(
        &features,
        &targets,
        SKSGDPenalty::L1,
        0.5,
        0.15,
        Some(start.clone()),
    );
    let elastic_l1 = fit_penalized(
        &features,
        &targets,
        SKSGDPenalty::ElasticNet,
        0.5,
        1.0,
        Some(start.clone()),
    );
    assert_eq!(l1.coefficients(), elastic_l1.coefficients());
    let l2 = fit_penalized(
        &features,
        &targets,
        SKSGDPenalty::L2,
        0.5,
        0.15,
        Some(start.clone()),
    );
    let elastic_l2 = fit_penalized(
        &features,
        &targets,
        SKSGDPenalty::ElasticNet,
        0.5,
        0.0,
        Some(start),
    );
    assert_eq!(l2.coefficients(), elastic_l2.coefficients());
}

#[test]
fn the_intercept_is_not_penalized() {
    let features = Array2::from_shape_vec((5, 1), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
    let targets = Array1::from_vec(vec![6.0, 8.0, 10.0, 12.0, 14.0]);
    let unpenalized = fit_penalized(&features, &targets, SKSGDPenalty::L2, 0.0, 0.15, None);
    let penalized = fit_penalized(&features, &targets, SKSGDPenalty::L2, 1.0, 0.15, None);
    assert_close_scalar(penalized.intercept(), 10.0, 0.05);
    assert_close_scalar(unpenalized.intercept(), 10.0, 0.05);
    assert!(penalized.coefficients()[0] < unpenalized.coefficients()[0]);
}
