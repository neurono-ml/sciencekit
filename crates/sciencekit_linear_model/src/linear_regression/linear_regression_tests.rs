//! Dense fit and prediction tests for `SKLinearRegression` (spec
//! `linear-regression`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKBackendKind, SKError, SKRegressorPredictor, SKSupervisedFit};

use super::core_implementation::SKLinearRegression;
use super::test_datasets::{
    SMALL_COEFFICIENTS, SMALL_INTERCEPT, assert_close_scalar, assert_close_with_tolerance,
    default_regressor, small_features, small_targets,
};

/// The fit recovers an exact line to epsilon.
#[test]
fn recovers_exact_coefficients_and_intercept() {
    let features = small_features();
    let targets = small_targets(&features);
    let model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close_with_tolerance(model.coefficients(), &SMALL_COEFFICIENTS, 1e-9);
    assert_close_scalar(model.intercept(), SMALL_INTERCEPT, 1e-9);
}

/// A full-rank design surfaces its rank and non-increasing singular values.
#[test]
fn surfaces_rank_and_singular_values() {
    let features = small_features();
    let targets = small_targets(&features);
    let model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    // Design width = 2 features + intercept = 3, full rank.
    assert_eq!(model.rank(), 3);
    assert_eq!(model.singular_values().len(), 3);
    assert!(
        model
            .singular_values()
            .windows(2)
            .all(|window| window[0] >= window[1]),
        "singular values must be non-increasing"
    );
}

/// With the intercept disabled the model reports a zero intercept.
#[test]
fn intercept_can_be_disabled() {
    let features = small_features();
    let targets = small_targets(&features);
    let estimator = SKLinearRegression::build()
        .include_intercept(false)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.intercept(), 0.0);
}

/// A rank-deficient design still fits, surfacing the reduced rank and returning
/// a solution that reproduces the targets.
#[test]
fn rank_deficiency_yields_a_minimum_norm_solution() {
    // Second column is exactly twice the first: the design is rank-deficient.
    let features = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
    let targets = array![1.0, 2.0, 3.0, 4.0];
    let model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(
        model.rank() < 3,
        "rank {} should be below the design width",
        model.rank()
    );
    let predictions = model.predict(features.view()).unwrap();
    assert_close_with_tolerance(&predictions, &[1.0, 2.0, 3.0, 4.0], 1e-9);
}

/// Prediction evaluates the learned line on new rows.
#[test]
fn predicts_new_rows() {
    let features = small_features();
    let targets = small_targets(&features);
    let model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let new_features = array![[3.0, 4.0], [5.0, 0.0]];
    let predictions = model.predict(new_features.view()).unwrap();
    // 3·3 − 2·4 + 0.5 = 1.5 ; 3·5 − 2·0 + 0.5 = 15.5
    assert_close_with_tolerance(&predictions, &[1.5, 15.5], 1e-9);
}

/// The estimator fits both supported floats.
#[test]
fn fits_both_supported_floats() {
    let features_f64 = small_features();
    let targets_f64 = small_targets(&features_f64);
    let model_f64 = default_regressor()
        .fit(features_f64.view(), targets_f64.view())
        .unwrap();
    assert_close_with_tolerance(model_f64.coefficients(), &SMALL_COEFFICIENTS, 1e-9);

    let features_f32 = features_f64.mapv(|value| value as f32);
    let targets_f32 = targets_f64.mapv(|value| value as f32);
    let model_f32 = default_regressor()
        .fit(features_f32.view(), targets_f32.view())
        .unwrap();
    for (value, expected) in model_f32.coefficients().iter().zip(SMALL_COEFFICIENTS) {
        assert!(
            (*value as f64 - expected).abs() < 1e-4,
            "f32 coefficient {value} differs from {expected}"
        );
    }
}

/// Empty input is rejected before any element is processed.
#[test]
fn rejects_empty_input() {
    let features: Array2<f64> = Array2::zeros((0, 2));
    let targets: Array1<f64> = Array1::zeros(0);
    let error = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Sparse input is unsupported here and rejected with a precise error.
#[test]
fn rejects_sparse_input() {
    use sprs::TriMat;
    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 0, 1.0_f64);
    triplets.add_triplet(1, 1, 1.0_f64);
    let sparse = triplets.to_csr();
    let targets = array![1.0, 2.0];
    let error = default_regressor()
        .fit(sparse.view(), targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

/// A target count disagreeing with the row count is rejected.
#[test]
fn rejects_target_length_mismatch() {
    let features = small_features();
    let targets = array![1.0, 2.0];
    let error = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Prediction on a column count differing from the fitted design is rejected.
#[test]
fn rejects_prediction_shape_mismatch() {
    let features = small_features();
    let targets = small_targets(&features);
    let model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let wrong = array![[1.0, 2.0, 3.0]];
    let error = model.predict(wrong.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The default backend is `Faer`.
#[test]
fn default_backend_is_faer() {
    let features = small_features();
    let targets = small_targets(&features);
    let model = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(model.backend(), SKBackendKind::Faer);
}

/// A non-default backend reaches the fitted model and still solves.
#[test]
fn backend_override_reaches_the_model() {
    let estimator = SKLinearRegression::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let features = small_features();
    let targets = small_targets(&features);
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_close_with_tolerance(model.coefficients(), &SMALL_COEFFICIENTS, 1e-9);
}

/// The fitted model is `Send + Sync`.
#[test]
fn model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<super::model::SKLinearRegressionModel<f64>>();
}

/// One fitted model serves concurrent predictions with identical results.
#[test]
fn model_serves_concurrent_predictions() {
    use std::sync::Arc;
    use std::thread;
    let features = small_features();
    let targets = small_targets(&features);
    let model = Arc::new(
        default_regressor()
            .fit(features.view(), targets.view())
            .unwrap(),
    );
    let expected = model.predict(features.view()).unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = features.clone();
            let expected = expected.clone();
            thread::spawn(move || {
                let predictions = model.predict(features.view()).unwrap();
                assert_eq!(predictions, expected);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}
