//! Out-of-core `fit_streaming` tests for `SKLasso` (spec `lasso`).
//!
//! This sibling owns the shared tight estimator and the source-provider tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::coordinate_descent::test_datasets::{
    SupervisedVecSource, assert_close, chunk_pairs, chunked_source, large_features, large_targets,
    orthogonal_design,
};
use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKReopenableSource, SKSupervisedFit};

use super::core_implementation::SKLasso;

/// A tight lasso over `f64`.
pub(super) fn lasso(alpha: f64) -> SKLasso<f64> {
    SKLasso::build()
        .alpha(alpha)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
}

/// Uneven batches converge to the dense fit.
#[test]
fn uneven_batches_match_the_dense_fit() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[1, 2, 1]);
    let streamed = lasso(0.5).fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// A reopenable provider takes exactly one pass: the retained design must not
/// be doubled.
#[test]
fn reopenable_provider_takes_one_pass_and_matches_the_dense_fit() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let batches = chunk_pairs(&features, &targets, &[1, 2, 1]);
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&factory_calls);
    let provider = SKReopenableSource::new(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        Ok(SupervisedVecSource::<f64>::new(batches.clone()))
    });
    let streamed = lasso(0.5).fit_streaming(provider).unwrap();
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// A single batch matches the dense fit.
#[test]
fn single_batch_matches_the_dense_fit() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[4]);
    let streamed = lasso(0.5).fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// An empty source is rejected with a structured error.
#[test]
fn empty_source_is_rejected() {
    let mut source: SupervisedVecSource<f64> = SupervisedVecSource::new(Vec::new());
    let error = lasso(0.5).fit_streaming(&mut source).unwrap_err();
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
    let error = lasso(0.5).fit_streaming(&mut source).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Streaming matches the dense fit at scale.
#[test]
fn streaming_matches_the_dense_fit_on_large_data() {
    let features = large_features::<f64>();
    let targets = large_targets(&features);
    let dense = lasso(0.25).fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[3_000, 4_000, 5_000]);
    let streamed = lasso(0.25).fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-5,
    );
    assert_eq!(streamed.row_count(), features.nrows() as u64);
}

/// An in-memory execution intent collects the batches and matches the dense fit.
#[test]
fn in_memory_intent_collects_and_matches_the_dense_fit() {
    use sciencekit_common::SKExecutionMode;

    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let estimator = SKLasso::<f64>::build()
        .alpha(0.5)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[2, 2]);
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// The streaming fit is independent of the batch order.
#[test]
fn streaming_is_independent_of_batch_order() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut permuted = SupervisedVecSource::new(vec![
        (
            features.slice(ndarray::s![2.., ..]).to_owned(),
            targets.slice(ndarray::s![2..]).to_owned(),
        ),
        (
            features.slice(ndarray::s![0..2, ..]).to_owned(),
            targets.slice(ndarray::s![0..2]).to_owned(),
        ),
    ]);
    let streamed = lasso(0.5).fit_streaming(&mut permuted).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}
