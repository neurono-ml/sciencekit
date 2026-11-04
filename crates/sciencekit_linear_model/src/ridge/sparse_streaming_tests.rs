//! Sparse streaming conformance tests for `SKRidge`.
//!
//! A sparse CSR batch stream accumulates the same centered cross products as the
//! dense stream and the dense `fit` on the concatenated data, counting each row
//! once, and never builds an `n × p` dense copy.

use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKExecutionMode, SKSupervisedFit};

use super::core_implementation::SKRidge;
use super::streaming_test_datasets::{
    assert_close_generic, assert_close_scalar_generic, chunked_source, small_features,
    small_targets,
};
use super::test_datasets::default_ridge;
use crate::sparse_streaming_test_datasets::{
    SparseSupervisedSource, chunk_pairs_csr, guard_matrix, guard_targets,
};

/// A sparse stream matches the dense stream and the dense concatenated fit.
#[test]
fn sparse_stream_matches_dense_stream_and_fit() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .expect("dense fit");
    let dense_stream = default_ridge()
        .fit_streaming(&mut chunked_source(&features, &targets, &[1, 2, 3]))
        .expect("dense stream");
    let sparse_stream = default_ridge()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[1, 2, 3],
        )))
        .expect("sparse stream");

    for model in [&dense, &dense_stream] {
        assert_close_generic(
            &sparse_stream.coefficient_for(0).to_vec(),
            &model.coefficient_for(0).to_vec(),
            1e-9,
        );
        assert_close_scalar_generic(sparse_stream.intercept_for(0), model.intercept_for(0), 1e-9);
    }
}

/// An explicit in-memory intent also accepts sparse batches.
#[test]
fn in_process_intent_accepts_sparse_batches() {
    let features = small_features::<f64>();
    let targets = small_targets(&features);
    let dense = default_ridge()
        .fit(features.view(), targets.view())
        .expect("dense fit");
    let collected = SKRidge::<f64>::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("estimator builds")
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[2, 4],
        )))
        .expect("sparse in-process stream");
    assert_close_generic(
        &collected.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-9,
    );
}

/// The sparse path never builds the dense `n × p` twin of the batch.
#[test]
fn sparse_stream_builds_no_dense_copy() {
    let rows = 250_000;
    let targets = guard_targets(rows);
    let mut source = SparseSupervisedSource::new(vec![(guard_matrix(rows, 64), targets)]);
    let streamed = default_ridge()
        .fit_streaming(&mut source)
        .expect("sparse stream");
    assert_eq!(streamed.coefficient_for(0).len(), 64);
}
