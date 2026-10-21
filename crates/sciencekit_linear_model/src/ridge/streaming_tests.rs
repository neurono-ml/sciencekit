//! Streaming fit tests (spec `ridge-regression`).

use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKExecutionMode, SKSupervisedFit};
use sciencekit_math::SKBackendSelection;

use super::builder::SKRidgeSolver;
use super::core_implementation::SKRidge;
use super::streaming_test_datasets::{
    SupervisedVecSource, assert_close_generic, assert_close_scalar_generic, chunked_source,
    large_features, large_targets, small_features, small_targets,
};
use super::test_datasets::default_ridge;

/// Uneven batches converge to the dense fit.
#[test]
fn uneven_batches_match_the_dense_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[1, 2, 3]);
    let streamed = default_ridge().fit_streaming(&mut source).unwrap();
    assert_close_generic(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
    assert_close_scalar_generic(streamed.intercept_for(0), dense.intercept_for(0), 1e-9);
}

/// A single batch matches the dense fit.
#[test]
fn single_batch_matches_the_dense_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[6]);
    let streamed = default_ridge().fit_streaming(&mut source).unwrap();
    assert_close_generic(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
}

/// An in-memory execution intent collects the batches and matches the dense fit.
#[test]
fn in_memory_intent_collects_and_matches_the_dense_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let estimator = SKRidge::<f64>::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[2, 4]);
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    assert_close_generic(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-12,
    );
}

/// An empty source is rejected with a structured error.
#[test]
fn empty_source_is_rejected() {
    let mut source: SupervisedVecSource<f64> = SupervisedVecSource::new(Vec::new());
    let error = default_ridge().fit_streaming(&mut source).unwrap_err();
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
    let error = default_ridge().fit_streaming(&mut source).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A multi-alpha fit streams one solution per value, matching the dense fit.
#[test]
fn multi_alpha_streaming_matches_the_dense_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let estimator = SKRidge::<f64>::build()
        .alpha(vec![0.5, 1.0, 2.0])
        .build()
        .unwrap();
    let dense = estimator.fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[2, 1, 3]);
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    assert_eq!(streamed.alphas(), &[0.5, 1.0, 2.0]);
    for index in 0..3 {
        assert_close_generic(
            &streamed.coefficient_for(index).to_vec(),
            &dense.coefficient_for(index).to_vec(),
            1e-9,
        );
        assert_close_scalar_generic(
            streamed.intercept_for(index),
            dense.intercept_for(index),
            1e-9,
        );
    }
}

/// A `SparseCg` intent falls back to the Gram solve and records `Cholesky`.
#[test]
fn sparse_cg_intent_falls_back_to_the_gram_solve() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let estimator = SKRidge::<f64>::build()
        .solver(SKRidgeSolver::SparseCg)
        .build()
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[3, 3]);
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    assert_eq!(streamed.solver(), SKRidgeSolver::Cholesky);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close_generic(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
}

/// The streaming fit honors a disabled intercept.
#[test]
fn streaming_honors_a_disabled_intercept() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let estimator = SKRidge::<f64>::build()
        .include_intercept(false)
        .build()
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[2, 2, 2]);
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    assert_close_scalar_generic(streamed.intercept_for(0), 0.0, 1e-12);
}

/// The streaming fit works on `f32`.
#[test]
fn streaming_fits_both_supported_floats() {
    streaming_matches_dense_for::<f32>(1e-3);
    streaming_matches_dense_for::<f64>(1e-9);
}

/// Streaming matches the dense fit at scale.
#[test]
fn streaming_matches_the_dense_fit_on_large_data() {
    let features = large_features::<f64>();
    let targets = large_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[3_000, 4_000, 5_000]);
    let streamed = default_ridge().fit_streaming(&mut source).unwrap();
    assert_close_generic(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-6,
    );
    assert_close_scalar_generic(streamed.intercept_for(0), dense.intercept_for(0), 1e-6);
    assert_eq!(streamed.row_count(), features.nrows() as u64);
}

/// Fit a small design densely and streamed and assert they agree, for any float.
fn streaming_matches_dense_for<F: SKBackendSelection + std::fmt::Debug>(tolerance: f64) {
    let features = small_features::<F>();
    let targets = small_targets(&features);
    let estimator = SKRidge::<F>::build().build().unwrap();
    let dense = estimator.fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[1, 2, 3]);
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    assert_close_generic(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        tolerance,
    );
    assert_close_scalar_generic(streamed.intercept_for(0), dense.intercept_for(0), tolerance);
}
