//! Structured-error tests for `SKElasticNet` (spec `elastic-net`).

use crate::coordinate_descent::test_datasets::one_feature;
use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKRegressorPredictor, SKSupervisedFit};

use super::core_implementation::SKElasticNet;

/// A default elastic net over `f64`.
fn default_net() -> SKElasticNet<f64> {
    SKElasticNet::build().build().unwrap()
}

/// Empty input is rejected before any element is processed.
#[test]
fn rejects_empty_design() {
    let features: Array2<f64> = Array2::zeros((0, 2));
    let targets: Array1<f64> = Array1::zeros(0);
    let error = default_net()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A target count disagreeing with the row count is rejected.
#[test]
fn rejects_target_length_mismatch() {
    let (features, _) = one_feature();
    let targets = ndarray::array![1.0, 2.0];
    let error = default_net()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A negative penalty is rejected at build time.
#[test]
fn rejects_negative_alpha() {
    let error = SKElasticNet::<f64>::build().alpha(-1.0).build().unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// An empty penalty sequence is rejected at build time.
#[test]
fn rejects_empty_alpha_sequence() {
    let error = SKElasticNet::<f64>::build()
        .alpha(Vec::<f64>::new())
        .build()
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// A non-positive tolerance is rejected at build time.
#[test]
fn rejects_zero_tolerance() {
    let error = SKElasticNet::<f64>::build().tolerance(0.0).build().unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "tolerance",
            ..
        }
    ));
}

/// A zero iteration budget is rejected at build time.
#[test]
fn rejects_zero_iteration_budget() {
    let error = SKElasticNet::<f64>::build()
        .maximum_number_of_iterations(0)
        .build()
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "maximum_number_of_iterations",
            ..
        }
    ));
}

/// An out-of-range `l1_ratio` is rejected at build time.
#[test]
fn rejects_out_of_range_l1_ratio() {
    for value in [-0.1_f64, 1.1] {
        let error = SKElasticNet::<f64>::build()
            .l1_ratio(value)
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "l1_ratio",
                ..
            }
        ));
    }
}

/// Prediction on a column count differing from the fitted design is rejected.
#[test]
fn rejects_prediction_shape_mismatch() {
    let (features, targets) = one_feature();
    let model = default_net()
        .fit(features.view(), targets.view())
        .unwrap();
    let wrong = ndarray::array![[1.0, 2.0]];
    let error = model.predict(wrong.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
