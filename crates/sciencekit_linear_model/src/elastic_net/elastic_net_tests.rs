//! Elastic-net fit, l1_ratio behaviour, optimality and prediction tests
//! (spec `elastic-net`).

use crate::coordinate_descent::test_datasets::{
    assert_close, assert_close_scalar, one_feature, orthogonal_design, zero_mean_feature,
};
use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKRegressorPredictor, SKSupervisedFit,
};

use crate::lasso::SKLasso;

use super::builder::SKElasticNetBuilder;
use super::core_implementation::SKElasticNet;

/// A tight, deterministic elastic net over `f64`.
fn elastic_net(alpha: f64, l1_ratio: f64) -> SKElasticNet<f64> {
    SKElasticNet::build()
        .alpha(alpha)
        .l1_ratio(l1_ratio)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .expect("elastic net builds")
}

/// A fresh default builder.
fn default_builder() -> SKElasticNetBuilder<f64> {
    SKElasticNet::build()
}

/// The default configuration is documented.
#[test]
fn default_configuration() {
    let estimator = SKElasticNet::<f64>::build().build().unwrap();
    assert_eq!(estimator.alphas(), &[1.0]);
    assert!((estimator.l1_ratio() - 0.5).abs() < 1e-12);
    assert!(estimator.include_intercept());
    assert!(!estimator.positive());
    assert!(estimator.screening());
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
}

/// `l1_ratio = 1` reproduces the lasso solution.
#[test]
fn l1_ratio_one_reproduces_lasso() {
    let (features, targets) = orthogonal_design();
    let lasso = SKLasso::<f64>::build()
        .alpha(0.5)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let net = elastic_net(0.5, 1.0)
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close(
        &lasso.coefficient_for(0).to_vec(),
        &net.coefficient_for(0).to_vec(),
        1e-6,
    );
}

/// `l1_ratio = 0` reproduces the pure-L2 closed form.
#[test]
fn l1_ratio_zero_reproduces_the_pure_l2_closed_form() {
    let (features, targets) = orthogonal_design();
    let alpha = 1.0;
    let model = elastic_net(alpha, 0.0)
        .fit(features.view(), targets.view())
        .unwrap();
    // Orthogonal mean-zero design with β = [2, −1, 0]: wⱼ = βⱼ / (1 + α).
    let expected = [2.0 / (1.0 + alpha), -1.0 / (1.0 + alpha), 0.0];
    assert_close(&model.coefficient_for(0).to_vec(), &expected, 1e-6);
}

/// The converged coefficients satisfy the elastic-net KKT conditions.
#[test]
fn kkt_conditions_hold() {
    let (features, targets) = orthogonal_design();
    let alpha = 0.5;
    let l1_ratio = 0.5;
    let model = elastic_net(alpha, l1_ratio)
        .fit(features.view(), targets.view())
        .unwrap();
    let coefficients = model.coefficient_for(0);
    let intercept = model.intercept_for(0);
    let residual = &targets - &features.dot(&coefficients) - intercept;
    let gradient = features.t().dot(&residual) / features.nrows() as f64;
    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio);
    for (column, &coefficient) in coefficients.iter().enumerate() {
        if coefficient.abs() > 1e-8 {
            let expected = l2 * coefficient + l1 * coefficient.signum();
            assert!((gradient[column] - expected).abs() < 1e-6);
        } else {
            assert!(gradient[column].abs() <= l1 + 1e-6);
        }
    }
}

/// A sequence of alphas yields one solution per value.
#[test]
fn alpha_sequence_yields_one_solution_per_value() {
    let (features, targets) = orthogonal_design();
    let estimator = SKElasticNet::build()
        .alpha(vec![0.25, 0.5, 1.5])
        .l1_ratio(0.5)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.alphas(), &[0.25, 0.5, 1.5]);
    assert_eq!(model.coefficients().nrows(), 3);
    assert_eq!(model.l1_ratio(), 0.5);
}

/// The penalty does not shrink the intercept.
#[test]
fn intercept_is_not_penalized() {
    let (features, targets) = zero_mean_feature();
    let weak = elastic_net(1e-9, 0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    let strong = elastic_net(10.0, 0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(strong.coefficient_for(0)[0].abs() < weak.coefficient_for(0)[0].abs());
    assert_close_scalar(weak.intercept_for(0), 5.0, 1e-6);
    assert_close_scalar(strong.intercept_for(0), 5.0, 1e-6);
}

/// Prediction evaluates the learned line on new rows.
#[test]
fn predicts_fitted_line() {
    let (features, targets) = one_feature();
    let model = elastic_net(1e-9, 0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    let predictions = model.predict(array![[4.0]].view()).unwrap();
    assert_close_scalar(predictions[0], 8.0, 1e-3);
}

/// Plain prediction refuses a multi-alpha model.
#[test]
fn plain_predict_refuses_multiple_alphas() {
    let (features, targets) = one_feature();
    let estimator = SKElasticNet::build()
        .alpha(vec![0.5, 1.0])
        .l1_ratio(0.5)
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
        .alpha(0.5)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}
