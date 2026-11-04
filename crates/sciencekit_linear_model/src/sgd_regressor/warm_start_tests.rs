//! Companion tests for the explicit warm start of the SGD regressor (spec
//! `stochastic-gradient-regressor`, decision 13).

use ndarray::Array1;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKRegressorPredictor, SKSupervisedFit};

use super::{SKSGDRegressor, SKSGDRegressorCoefficients};
use crate::sgd::SKConvergencePolicy;
use crate::test_datasets::{assert_close, regression_design};

#[test]
fn a_warm_start_without_intercept_starts_from_zero() {
    let (features, targets) = regression_design();
    let source = SKSGDRegressorCoefficients::new(Array1::from_vec(vec![1.0, 1.0]).view(), 42.0);
    let model = SKSGDRegressor::<f64>::build()
        .include_intercept(false)
        .warm_start_from(source)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(3)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.intercept(), 0.0);
    let predicted = model.predict(features.view()).unwrap();
    let expected = features.dot(&model.coefficients());
    assert_close(
        predicted.as_slice().unwrap(),
        expected.as_slice().unwrap(),
        1e-12,
    );
}
