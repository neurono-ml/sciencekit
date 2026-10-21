//! Dense lasso fit, sparsity, optimality and prediction tests (spec `lasso`).

use crate::coordinate_descent::test_datasets::{
    assert_close, assert_close_scalar, one_feature, orthogonal_design, soft_threshold,
    zero_mean_feature,
};
use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKRegressorPredictor, SKSupervisedFit,
};

use super::builder::SKLassoBuilder;
use super::core_implementation::SKLasso;

/// A tight, deterministic lasso over `f64`.
fn lasso(alpha: f64) -> SKLasso<f64> {
    SKLasso::build()
        .alpha(alpha)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .expect("lasso builds")
}

/// A fresh default builder.
fn default_builder() -> SKLassoBuilder<f64> {
    SKLasso::build()
}

/// The default configuration is documented.
#[test]
fn default_configuration() {
    let estimator = SKLasso::<f64>::build().build().unwrap();
    assert_eq!(estimator.alphas(), &[1.0]);
    assert!(estimator.include_intercept());
    assert!(!estimator.positive());
    assert!(estimator.screening());
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.maximum_number_of_iterations(), 1000);
    assert!((estimator.tolerance() - 1e-4).abs() < 1e-12);
}

/// On the orthogonal design the coefficients match `S(βⱼ, α)`.
#[test]
fn orthogonal_design_matches_the_closed_form() {
    let (features, targets) = orthogonal_design();
    let model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    assert_close(&model.coefficient_for(0).to_vec(), &[1.5, -0.5, 0.0], 1e-6);
    assert_close_scalar(model.intercept_for(0), 0.0, 1e-8);
}

/// A feature the penalty excludes is driven to an exact zero.
#[test]
fn irrelevant_feature_is_exactly_zero() {
    let (features, targets) = orthogonal_design();
    let model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.coefficient_for(0)[2], 0.0);
}

/// The converged coefficients satisfy the lasso KKT conditions.
#[test]
fn kkt_conditions_hold() {
    let (features, targets) = orthogonal_design();
    let alpha = 0.5;
    let model = lasso(alpha).fit(features.view(), targets.view()).unwrap();
    let coefficients = model.coefficient_for(0);
    let intercept = model.intercept_for(0);
    let residual = &targets - &features.dot(&coefficients) - intercept;
    let gradient = features.t().dot(&residual) / features.nrows() as f64;
    for (column, &coefficient) in coefficients.iter().enumerate() {
        if coefficient.abs() > 1e-8 {
            assert!((gradient[column].abs() - alpha).abs() < 1e-6);
        } else {
            assert!(gradient[column].abs() <= alpha + 1e-6);
        }
    }
}

/// A sequence of alphas yields one solution per value, in order.
#[test]
fn alpha_sequence_yields_one_solution_per_value() {
    let (features, targets) = orthogonal_design();
    let estimator = SKLasso::build()
        .alpha(vec![0.25, 0.5, 1.5])
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.alphas(), &[0.25, 0.5, 1.5]);
    assert_eq!(model.coefficients().nrows(), 3);
    let expected = [
        soft_threshold(2.0, 0.25),
        soft_threshold(2.0, 0.5),
        soft_threshold(2.0, 1.5),
    ];
    for (row, &value) in expected.iter().enumerate() {
        assert_close_scalar(model.coefficient_for(row)[0], value, 1e-6);
    }
}

/// The penalty does not shrink the intercept.
#[test]
fn intercept_is_not_penalized() {
    let (features, targets) = zero_mean_feature();
    let weak = lasso(1e-9).fit(features.view(), targets.view()).unwrap();
    let strong = lasso(10.0).fit(features.view(), targets.view()).unwrap();
    assert!(strong.coefficient_for(0)[0].abs() < weak.coefficient_for(0)[0].abs());
    assert_close_scalar(weak.intercept_for(0), 5.0, 1e-6);
    assert_close_scalar(strong.intercept_for(0), 5.0, 1e-6);
}

/// With the intercept disabled the model reports a zero intercept.
#[test]
fn intercept_can_be_disabled() {
    let (features, targets) = one_feature();
    let estimator = SKLasso::build()
        .alpha(1.0)
        .include_intercept(false)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.intercept_for(0), 0.0);
    assert!(!model.include_intercept());
}

/// The positive constraint clamps a negative coefficient at zero.
#[test]
fn positive_constraint_clamps() {
    let (features, targets) = orthogonal_design();
    let estimator = SKLasso::build()
        .alpha(0.5)
        .positive(true)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert!(model.positive());
    assert_close(&model.coefficient_for(0).to_vec(), &[1.5, 0.0, 0.0], 1e-6);
}

/// Prediction evaluates the learned line on new rows.
#[test]
fn predicts_fitted_line() {
    let (features, targets) = one_feature();
    let model = lasso(1e-9).fit(features.view(), targets.view()).unwrap();
    let predictions = model.predict(array![[4.0]].view()).unwrap();
    assert_close_scalar(predictions[0], 8.0, 1e-4);
}

/// Plain prediction refuses a multi-alpha model.
#[test]
fn plain_predict_refuses_multiple_alphas() {
    let (features, targets) = one_feature();
    let estimator = SKLasso::build()
        .alpha(vec![0.5, 1.0])
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    let error = model.predict(features.view()).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// A non-default backend reaches the fitted model.
#[test]
fn non_default_backend_reaches_model() {
    let (features, targets) = one_feature();
    let estimator = default_builder()
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
    let model = SKLasso::<f32>::build()
        .alpha(1.0f32)
        .include_intercept(false)
        .tolerance(1e-5)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!((model.coefficient_for(0)[0] - 25.0 / 14.0).abs() < 1e-3);
}

/// The fitted model is `Send + Sync`.
#[test]
fn model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<super::model::SKLassoModel<f64>>();
}
