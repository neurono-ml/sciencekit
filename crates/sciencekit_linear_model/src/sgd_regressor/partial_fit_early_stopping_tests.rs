//! Companion tests for incremental learning after an early-stopped SGD
//! regressor fit (spec `stochastic-gradient-solver`, decision 7).

use ndarray::{Array1, Array2};
use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::SKSGDRegressor;
use super::partial_fit_tests::mean_squared;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate};

fn anti_correlated_design() -> (Array2<f64>, Array1<f64>) {
    let values = [
        -1.0, -0.75, -0.5, -0.25, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
        3.25, 3.5, 3.75, 4.0,
    ];
    let features = Array2::from_shape_vec((values.len(), 1), values.to_vec()).unwrap();
    let targets = Array1::from_iter(
        values
            .iter()
            .enumerate()
            .map(|(row, &value)| if row < 5 { -2.0 * value } else { 2.0 * value }),
    );
    (features, targets)
}

#[test]
fn partial_fit_after_budget_exhaustion_continues_from_the_reported_iterate() {
    let (features, targets) = anti_correlated_design();
    // A budget smaller than the patience stops the fit at the budget, not at
    // the patience: the model still reports the best-validation iterate, and
    // the retained state must agree with it before any continuation.
    let increment = features.slice(ndarray::s![0..5, ..]).to_owned();
    let increment_targets = targets.slice(ndarray::s![0..5]).to_owned();
    let mut model = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(1e-4)
        .alpha(0.0)
        .shuffle(false)
        .early_stopping(true)
        .validation_fraction(0.25)
        .number_of_epochs_without_improvement(10)
        .maximum_number_of_iterations(2)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(!model.converged());
    assert_eq!(model.coefficients(), model.state.weights.view());
    let before = mean_squared(&model, &increment, &increment_targets);
    model
        .partial_fit(increment.view(), increment_targets.view())
        .unwrap();
    let after = mean_squared(&model, &increment, &increment_targets);
    assert!(
        after <= before + 1e-12,
        "the continuation regressed from {before} to {after}"
    );
}

#[test]
fn partial_fit_after_early_stopping_continues_from_the_best_iterate() {
    let (features, targets) = anti_correlated_design();
    // The validation prefix is the slice early stopping keeps the best iterate
    // on; continuing on it exposes whether the state starts from the reported
    // (best) iterate or from the discarded last one.
    let increment = features.slice(ndarray::s![0..5, ..]).to_owned();
    let increment_targets = targets.slice(ndarray::s![0..5]).to_owned();
    let mut model = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(1e-4)
        .alpha(0.0)
        .shuffle(false)
        .early_stopping(true)
        .validation_fraction(0.25)
        .number_of_epochs_without_improvement(1)
        .maximum_number_of_iterations(10)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(model.converged());
    let before = mean_squared(&model, &increment, &increment_targets);
    model
        .partial_fit(increment.view(), increment_targets.view())
        .unwrap();
    let after = mean_squared(&model, &increment, &increment_targets);
    assert!(
        after <= before + 1e-12,
        "the continuation regressed from {before} to {after}"
    );
}
