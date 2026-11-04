//! Companion tests for incremental learning of the fitted SGD regressor (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKRegressorPredictor, SKSupervisedFit};

use super::SKSGDRegressor;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate};
use crate::test_datasets::{assert_close, regression_design};

fn constant_one_epoch(alpha: f64, average: bool) -> SKSGDRegressor<f64> {
    SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(alpha)
        .shuffle(false)
        .average(average)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
}

fn mean_squared(
    model: &super::SKSGDRegressorModel<f64>,
    features: &Array2<f64>,
    targets: &Array1<f64>,
) -> f64 {
    let predicted = model.predict(features.view()).unwrap();
    predicted
        .iter()
        .zip(targets.iter())
        .map(|(value, target)| (value - target) * (value - target))
        .sum::<f64>()
        / targets.len() as f64
}

#[test]
fn partial_fit_matches_the_concatenated_fit() {
    let (features, targets) = regression_design();
    let first = features.slice(ndarray::s![0..3, ..]).to_owned();
    let first_targets = targets.slice(ndarray::s![0..3]).to_owned();
    let rest = features.slice(ndarray::s![3.., ..]).to_owned();
    let rest_targets = targets.slice(ndarray::s![3..]).to_owned();
    let mut incremental = constant_one_epoch(0.0, false)
        .fit(first.view(), first_targets.view())
        .unwrap();
    incremental
        .partial_fit(rest.view(), rest_targets.view())
        .unwrap();
    let concatenated = constant_one_epoch(0.0, false)
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close(
        incremental.coefficients().as_slice().unwrap(),
        concatenated.coefficients().as_slice().unwrap(),
        1e-12,
    );
    assert_close(
        &[incremental.intercept()],
        &[concatenated.intercept()],
        1e-12,
    );
}

#[test]
fn partial_fit_continues_the_schedule_and_the_average() {
    let (features, targets) = regression_design();
    let mut model = constant_one_epoch(0.0, true)
        .fit(features.view(), targets.view())
        .unwrap();
    let before_step = model.state.step;
    let before_average = model.state.average_count();
    let before_epochs = model.epochs_taken();
    model.partial_fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.state.step, before_step + features.nrows());
    assert!(model.state.average_count() > before_average);
    assert_eq!(model.epochs_taken(), before_epochs + 1);
    assert!(!model.converged());
}

#[test]
fn partial_fit_does_not_increase_the_objective() {
    let (features, targets) = regression_design();
    let mut model = constant_one_epoch(0.0, false)
        .fit(features.view(), targets.view())
        .unwrap();
    let before = mean_squared(&model, &features, &targets);
    model.partial_fit(features.view(), targets.view()).unwrap();
    let after = mean_squared(&model, &features, &targets);
    assert!(
        after <= before + 1e-9,
        "the partial fit raised the objective from {before} to {after}"
    );
}

/// A design whose validation prefix is anti-correlated with the training
/// remainder, so early stopping is guaranteed to trigger after one epoch.
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

#[test]
fn a_column_mismatch_is_rejected() {
    let (features, targets) = regression_design();
    let mut model = constant_one_epoch(0.0, false)
        .fit(features.view(), targets.view())
        .unwrap();
    let wider = Array2::<f64>::zeros((5, 3));
    let error = model.partial_fit(wider.view(), targets.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn a_target_count_mismatch_is_rejected() {
    let (features, targets) = regression_design();
    let mut model = constant_one_epoch(0.0, false)
        .fit(features.view(), targets.view())
        .unwrap();
    let short = Array1::from_vec(vec![1.0, 2.0]);
    let error = model
        .partial_fit(features.view(), short.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn an_f32_partial_fit_continues() {
    let features = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
    let targets = array![3.0f32, 6.0, 9.0, 12.0, 15.0];
    let mut model = SKSGDRegressor::<f32>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.01)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    model.partial_fit(features.view(), targets.view()).unwrap();
    assert!(model.coefficients().iter().all(|value| value.is_finite()));
    assert!(model.coefficients()[0] > 0.0);
}
