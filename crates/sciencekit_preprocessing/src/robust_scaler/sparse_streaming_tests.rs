//! Sparse streaming conformance tests for `SKRobustScaler`.
//!
//! The per-column sketches see the explicit values plus the implicit-zero
//! multiplicity, so a sparse stream approximates the dense stream and the exact
//! dense `fit` within the documented sketch tolerance. The guarded test streams
//! a shape whose dense twin is impractical for the test budget; it proves the
//! stream completes there, not that no dense copy exists.

use sciencekit_common::{SKBuilder, SKExecutionMode, SKUnsupervisedFit};

use super::core_implementation::SKRobustScaler;
use super::test_datasets::{
    SKETCH_RELATIVE_TOLERANCE, assert_close_with_tolerance, chunked_source, default_scaler,
};
use crate::sparse_streaming_test_datasets::{
    SparseVecSource, chunk_csr, guard_matrix, zeros_rich_large_features,
};

/// A sparse stream approximates the dense stream and the dense concatenated fit.
#[test]
fn sparse_stream_matches_dense_stream_and_fit() {
    let features = zeros_rich_large_features(1_200);
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let dense_stream = default_scaler()
        .fit_streaming(&mut chunked_source(&features, &[300, 400, 500]))
        .expect("dense stream succeeds");
    let sparse_stream = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(
            &features,
            &[300, 400, 500],
        )))
        .expect("sparse stream succeeds");

    assert_eq!(sparse_stream.sample_count(), reference.sample_count());
    assert_eq!(sparse_stream.sample_count(), dense_stream.sample_count());
    for (actual, expected) in [
        (sparse_stream.center(), reference.center()),
        (sparse_stream.scale(), reference.scale()),
        (sparse_stream.center(), dense_stream.center()),
        (sparse_stream.scale(), dense_stream.scale()),
    ] {
        assert_close_with_tolerance(actual, &expected.to_vec(), SKETCH_RELATIVE_TOLERANCE);
    }
}

/// A mostly-implicit column has its median and interquantile range driven to
/// zero, exactly as the dense fit sees it.
#[test]
fn implicit_zeros_drive_the_sketch_quantiles() {
    let features = ndarray::array![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]];
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let sparse_stream = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[1, 3])))
        .expect("sparse stream succeeds");
    assert_close_with_tolerance(
        sparse_stream.center(),
        &reference.center().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
    assert_eq!(sparse_stream.center()[0], 0.0);
}

/// The stream completes at a shape whose dense twin is impractical for the test
/// budget.
#[test]
fn sparse_stream_handles_a_shape_beyond_the_test_budget() {
    let batches = vec![
        guard_matrix(100_000, 1_000),
        guard_matrix(100_000, 1_000),
        guard_matrix(100_000, 1_000),
    ];
    let streamed = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(batches))
        .expect("sparse stream succeeds");
    assert_eq!(streamed.sample_count(), 300_000);
    assert_eq!(streamed.center().len(), 1_000);
}

/// An explicit in-memory intent also accepts sparse batches.
#[test]
fn in_process_intent_accepts_sparse_batches() {
    let features = zeros_rich_large_features(300);
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let collected = SKRobustScaler::<f64>::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("scaler builds")
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[100, 200])))
        .expect("sparse in-process stream succeeds");
    assert_close_with_tolerance(
        collected.center(),
        &reference.center().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
}
