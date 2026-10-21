//! Dense fit, multi-alpha model shape and prediction tests for `SKRidge`
//! (spec `ridge-regression`).

use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKRegressorPredictor, SKSupervisedFit,
};

use super::builder::SKRidgeSolver;
use super::core_implementation::SKRidge;
use super::test_datasets::{
    assert_close, assert_close_scalar, default_ridge, one_feature, zero_mean_feature,
};

/// Without an intercept the closed-form slope is `Σxy / (Σx² + α)`.
#[test]
fn hand_checkable_ridge_solution_without_intercept() {
    let (features, targets) = one_feature();
    let estimator = SKRidge::build()
        .alpha(2.0)
        .include_intercept(false)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    // Σxy = 28, Σx² = 14, α = 2 → 28 / 16 = 1.75.
    assert_close(&model.coefficient_for(0).to_vec(), &[1.75], 1e-9);
    assert_close_scalar(model.intercept_for(0), 0.0, 1e-12);
    assert_eq!(model.alphas(), &[2.0]);
}

/// A scalar `alpha` yields exactly one solution.
#[test]
fn scalar_alpha_yields_one_solution() {
    let (features, targets) = one_feature();
    let model = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.coefficients().nrows(), 1);
    assert_eq!(model.intercepts().len(), 1);
}

/// A sequence yields one solution per value, in the configured order.
#[test]
fn alpha_sequence_yields_one_solution_per_value() {
    let (features, targets) = one_feature();
    let estimator = SKRidge::build().alpha(vec![0.0, 1.0, 4.0]).build().unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.alphas(), &[0.0, 1.0, 4.0]);
    assert_eq!(model.coefficients().nrows(), 3);
    assert_eq!(model.intercepts().len(), 3);
    // Growing the penalty shrinks the slope.
    assert!(model.coefficient_for(0)[0] > model.coefficient_for(1)[0]);
    assert!(model.coefficient_for(1)[0] > model.coefficient_for(2)[0]);
}

/// The penalty shrinks the slope but leaves the unpenalized intercept intact.
#[test]
fn intercept_is_not_penalized() {
    let (features, targets) = zero_mean_feature();
    let weak = SKRidge::build()
        .alpha(1e-9)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let strong = SKRidge::build()
        .alpha(10.0)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(strong.coefficient_for(0)[0].abs() < weak.coefficient_for(0)[0].abs());
    assert_close_scalar(weak.intercept_for(0), 5.0, 1e-6);
    assert_close_scalar(strong.intercept_for(0), 5.0, 1e-6);
}

/// With the intercept disabled the model reports a zero intercept.
#[test]
fn intercept_can_be_disabled() {
    let (features, targets) = one_feature();
    let estimator = SKRidge::build().include_intercept(false).build().unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.intercept_for(0), 0.0);
    assert!(!model.include_intercept());
}

/// Prediction evaluates the learned line on new rows.
#[test]
fn predicts_fitted_line() {
    let (features, targets) = one_feature();
    let model = SKRidge::build()
        .alpha(1e-9)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let predictions = model.predict(array![[4.0]].view()).unwrap();
    assert_close_scalar(predictions[0], 8.0, 1e-6);
}

/// Plain prediction refuses a multi-alpha model.
#[test]
fn plain_predict_refuses_multiple_alphas() {
    let (features, targets) = one_feature();
    let model = SKRidge::build()
        .alpha(vec![1.0, 2.0])
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let error = model.predict(features.view()).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// The explicit per-alpha prediction selects the requested coefficient row.
#[test]
fn predict_with_alpha_index_selects_a_row() {
    let (features, targets) = one_feature();
    let model = SKRidge::build()
        .alpha(vec![0.0, 10.0])
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let exact = model.predict_with_alpha_index(0, features.view()).unwrap();
    let shrunk = model.predict_with_alpha_index(1, features.view()).unwrap();
    assert_eq!(exact.len(), features.nrows());
    // The larger penalty shrinks the slope, so the first row's response differs.
    assert!((exact[0] - shrunk[0]).abs() > 1e-6);
}

/// The `Automatic` solver resolves to `Cholesky` on a dense full-rank fit.
#[test]
fn default_solver_resolves_to_cholesky() {
    let (features, targets) = one_feature();
    let model = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.solver(), SKRidgeSolver::Cholesky);
    assert_eq!(model.backend(), SKBackendKind::Faer);
}

/// A non-default backend reaches the fitted model.
#[test]
fn non_default_backend_reaches_model() {
    let (features, targets) = one_feature();
    let estimator = SKRidge::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}

/// The estimator fits both supported floats.
#[test]
fn fits_both_supported_floats() {
    let features = array![[1.0f32], [2.0], [3.0]];
    let targets = array![2.0f32, 4.0, 6.0];
    let model = SKRidge::<f32>::build()
        .alpha(2.0f32)
        .include_intercept(false)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!((model.coefficient_for(0)[0] - 1.75).abs() < 1e-4);
}

/// The default configuration is documented and automatic.
#[test]
fn default_configuration() {
    let estimator = SKRidge::<f64>::build().build().unwrap();
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
    assert!(estimator.include_intercept());
    assert_eq!(estimator.solver(), SKRidgeSolver::Automatic);
    assert_eq!(estimator.alphas(), &[1.0]);
    assert_eq!(estimator.maximum_number_of_iterations(), 1000);
}

/// The fitted model is `Send + Sync`.
#[test]
fn model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<super::model::SKRidgeModel<f64>>();
}
