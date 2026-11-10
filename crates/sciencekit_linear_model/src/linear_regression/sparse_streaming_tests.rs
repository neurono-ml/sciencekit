//! Sparse streaming conformance tests for `SKLinearRegression`.
//!
//! A sparse CSR batch stream accumulates the same augmented Gram state as the
//! dense stream and the dense `fit` on the concatenated data, counting each row
//! once. The guarded test streams a shape whose dense twin is impractical for
//! the test budget; it proves the stream completes there, not that no dense copy
//! exists.

use sciencekit_common::{SKBuilder, SKExecutionMode, SKSupervisedFit};

use super::core_implementation::SKLinearRegression;
use super::test_datasets::{
    assert_close_scalar, assert_close_with_tolerance, chunked_source, default_regressor,
    small_features, small_targets,
};
use crate::sparse_streaming_test_datasets::{
    SparseSupervisedSource, chunk_pairs_csr, guard_matrix, guard_targets,
};

/// A sparse stream matches the dense stream and the dense concatenated fit.
#[test]
fn sparse_stream_matches_dense_stream_and_fit() {
    let features = small_features();
    let targets = small_targets(&features);
    let reference = default_regressor()
        .fit(features.view(), targets.view())
        .expect("dense fit");
    let dense_stream = default_regressor()
        .fit_streaming(&mut chunked_source(&features, &targets, &[1, 1, 2]))
        .expect("dense stream");
    let sparse_stream = default_regressor()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[1, 1, 2],
        )))
        .expect("sparse stream");

    assert_eq!(sparse_stream.row_count(), reference.row_count());
    assert_eq!(sparse_stream.row_count(), dense_stream.row_count());
    for model in [&reference, &dense_stream] {
        assert_close_with_tolerance(
            sparse_stream.coefficients(),
            &model.coefficients().to_vec(),
            1e-9,
        );
        assert_close_scalar(sparse_stream.intercept(), model.intercept(), 1e-9);
    }
}

/// An explicit in-memory intent also accepts sparse batches.
#[test]
fn in_process_intent_accepts_sparse_batches() {
    let features = small_features();
    let targets = small_targets(&features);
    let reference = default_regressor()
        .fit(features.view(), targets.view())
        .expect("dense fit");
    let collected = SKLinearRegression::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("estimator builds")
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[2, 2],
        )))
        .expect("sparse in-process stream");
    assert_close_with_tolerance(
        collected.coefficients(),
        &reference.coefficients().to_vec(),
        1e-9,
    );
}

/// The stream completes at a shape whose dense twin is impractical for the test
/// budget.
#[test]
fn sparse_stream_handles_a_shape_beyond_the_test_budget() {
    let rows = 250_000;
    let targets = guard_targets(rows);
    let mut source = SparseSupervisedSource::<f64>::new(vec![(guard_matrix(rows, 64), targets)]);
    let streamed = default_regressor()
        .fit_streaming(&mut source)
        .expect("sparse stream");
    assert_eq!(streamed.row_count(), rows as u64);
    assert_eq!(streamed.coefficients().len(), 64);
}

/// A sparse incremental update accumulates the same Gram state as a dense one.
#[test]
fn sparse_streaming_partial_fit_matches_the_dense_fit() {
    use sciencekit_common::SKRegressorPredictor;

    let features = small_features();
    let targets = small_targets(&features);
    let reference = default_regressor()
        .fit(&features, targets.view())
        .expect("fit succeeds");
    let chunks = chunk_pairs_csr(&features, &targets, &[1, 1, 2]);
    let mut model = default_regressor()
        .fit(&chunks[0].0.to_dense(), chunks[0].1.view())
        .expect("dense fit succeeds");
    for (batch, batch_targets) in &chunks[1..] {
        model
            .partial_fit(batch.view(), batch_targets.view())
            .expect("sparse partial_fit succeeds");
    }
    let predicted = model.predict(features.view()).expect("predict succeeds");
    let expected = reference
        .predict(features.view())
        .expect("predict succeeds");
    for (actual, expected) in predicted.iter().zip(expected.iter()) {
        assert_close_scalar(*actual, *expected, 1e-9);
    }
}
