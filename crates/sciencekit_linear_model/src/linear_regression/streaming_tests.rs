//! Streaming fit tests (spec `linear-regression`).

use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKExecutionMode, SKSupervisedFit};

use super::core_implementation::SKLinearRegression;
use super::test_datasets::{
    SMALL_COEFFICIENTS, SupervisedVecSource, assert_close_scalar, assert_close_with_tolerance,
    chunked_source, default_regressor, large_features, large_targets, small_features,
    small_targets,
};

/// Uneven batches converge to the dense fit.
#[test]
fn uneven_batches_match_the_dense_fit() {
    let features = small_features();
    let targets = small_targets(&features);
    let dense = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[1, 3]);
    let streamed = default_regressor().fit_streaming(&mut source).unwrap();
    assert_close_with_tolerance(
        streamed.coefficients(),
        dense.coefficients().as_slice().unwrap(),
        1e-9,
    );
    assert_close_scalar(streamed.intercept(), dense.intercept(), 1e-9);
}

/// A single batch matches the dense fit.
#[test]
fn single_batch_matches_the_dense_fit() {
    let features = small_features();
    let targets = small_targets(&features);
    let dense = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[4]);
    let streamed = default_regressor().fit_streaming(&mut source).unwrap();
    assert_close_with_tolerance(
        streamed.coefficients(),
        dense.coefficients().as_slice().unwrap(),
        1e-9,
    );
}

/// An in-memory execution intent collects the batches and matches the dense
/// fit exactly.
#[test]
fn in_memory_intent_collects_and_matches_the_dense_fit() {
    let features = small_features();
    let targets = small_targets(&features);
    let dense = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let estimator = SKLinearRegression::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[2, 2]);
    let model = estimator.fit_streaming(&mut source).unwrap();
    assert_close_with_tolerance(
        model.coefficients(),
        dense.coefficients().as_slice().unwrap(),
        1e-12,
    );
}

/// An empty source is rejected with a structured error.
#[test]
fn empty_source_is_rejected() {
    let mut source: SupervisedVecSource<f64> = SupervisedVecSource::new(Vec::new());
    let error = default_regressor().fit_streaming(&mut source).unwrap_err();
    assert!(matches!(error, SKError::Conversion(_)));
}

/// Batches with differing column counts are rejected.
#[test]
fn column_mismatch_is_rejected() {
    let batches = vec![
        (array![[1.0, 2.0], [3.0, 4.0]], array![1.0, 2.0]),
        (array![[1.0, 2.0, 3.0]], array![1.0]),
    ];
    let mut source = SupervisedVecSource::new(batches);
    let error = default_regressor().fit_streaming(&mut source).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The streaming fit honors a disabled intercept.
#[test]
fn streaming_honors_a_disabled_intercept() {
    let features = small_features();
    let targets = small_targets(&features);
    let estimator = SKLinearRegression::build()
        .include_intercept(false)
        .build()
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[2, 2]);
    let model = estimator.fit_streaming(&mut source).unwrap();
    assert_eq!(model.intercept(), 0.0);
}

/// The streaming fit works on `f32`.
#[test]
fn streaming_fits_both_supported_floats() {
    let features = small_features().mapv(|value| value as f32);
    let targets = small_targets(&small_features()).mapv(|value| value as f32);
    let mut source = chunked_source(&features, &targets, &[1, 1, 2]);
    let model = default_regressor().fit_streaming(&mut source).unwrap();
    for (value, expected) in model.coefficients().iter().zip(SMALL_COEFFICIENTS) {
        assert!(
            (*value as f64 - expected).abs() < 1e-3,
            "f32 streaming coefficient {value} differs from {expected}"
        );
    }
}

/// Streaming matches the dense fit at scale.
#[test]
fn streaming_matches_the_dense_fit_on_large_data() {
    let features = large_features();
    let targets = large_targets(&features);
    let dense = default_regressor()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[5_000, 7_000, 8_000]);
    let streamed = default_regressor().fit_streaming(&mut source).unwrap();
    assert_close_with_tolerance(
        streamed.coefficients(),
        dense.coefficients().as_slice().unwrap(),
        1e-6,
    );
    assert_close_scalar(streamed.intercept(), dense.intercept(), 1e-6);
    assert_eq!(streamed.row_count(), features.nrows() as u64);
}
