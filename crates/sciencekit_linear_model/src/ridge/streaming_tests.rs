//! Streaming fit tests (spec `ridge-regression`).
//!
//! Core source-provider equivalence: uneven batches, a single batch, the
//! one-pass provider and the rejected inputs.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKExecutionMode, SKReopenableSource, SKSupervisedFit};

use super::core_implementation::SKRidge;
use super::streaming_test_datasets::{
    SupervisedVecSource, assert_close_generic, assert_close_scalar_generic, chunk_pairs,
    chunked_source, small_features, small_targets,
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

/// A reopenable provider takes exactly one pass: the centered Gram accumulator
/// must not be doubled.
#[test]
fn reopenable_provider_takes_one_pass_and_matches_the_dense_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .unwrap();
    let batches = chunk_pairs(&features, &targets, &[2, 4]);
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&factory_calls);
    let provider = SKReopenableSource::new(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        Ok(SupervisedVecSource::<f64>::new(batches.clone()))
    });
    let streamed = default_ridge().fit_streaming(provider).unwrap();
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
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
