//! Sparse streaming conformance tests for `SKStandardScaler`.
//!
//! A sparse CSR stream must learn the same moments as the dense stream and the
//! dense `fit` on the concatenated data: implicit zeros participate in every
//! column's count, mean and M2, and no `n × p` dense copy is ever built.

use sciencekit_common::{SKBuilder, SKExecutionMode, SKUnsupervisedFit};

use super::core_implementation::SKStandardScaler;
use super::test_datasets::{chunked_source, default_scaler, small_features};
use crate::sparse_streaming_test_datasets::{
    SparseVecSource, assert_close_sparse, chunk_csr, guard_matrix, zeros_rich_features,
};

/// A sparse stream matches the dense stream and the dense concatenated fit.
#[test]
fn sparse_stream_matches_dense_stream_and_fit() {
    let features = zeros_rich_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let dense_stream = default_scaler()
        .fit_streaming(&mut chunked_source(&features, &[2, 3, 3]))
        .expect("dense stream succeeds");
    let sparse_stream = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 3, 3])))
        .expect("sparse stream succeeds");

    assert_eq!(sparse_stream.sample_count(), reference.sample_count());
    assert_eq!(sparse_stream.sample_count(), dense_stream.sample_count());
    assert_close_sparse(
        &sparse_stream.mean().to_vec(),
        &reference.mean().to_vec(),
        1e-9,
    );
    assert_close_sparse(
        &sparse_stream.mean().to_vec(),
        &dense_stream.mean().to_vec(),
        1e-9,
    );
    assert_close_sparse(
        &sparse_stream.scale().to_vec(),
        &reference.scale().to_vec(),
        1e-9,
    );
    assert_close_sparse(
        &sparse_stream.scale().to_vec(),
        &dense_stream.scale().to_vec(),
        1e-9,
    );
}

/// Implicit zeros count as observed zeros: a column with no stored entry has
/// mean zero and unit scale, and a mostly-zero column keeps its dense mean.
#[test]
fn implicit_zeros_participate_in_the_column_statistics() {
    let features = ndarray::array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let sparse_stream = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[1, 3])))
        .expect("sparse stream succeeds");
    assert_close_sparse(
        &sparse_stream.mean().to_vec(),
        &reference.mean().to_vec(),
        1e-12,
    );
    assert_close_sparse(
        &sparse_stream.scale().to_vec(),
        &reference.scale().to_vec(),
        1e-12,
    );
    assert_eq!(sparse_stream.mean()[1], 0.0);
}

/// The sparse path never builds the dense `n × p` twin of the batch.
#[test]
fn sparse_stream_builds_no_dense_copy() {
    let batches = vec![
        guard_matrix(100_000, 100_000),
        guard_matrix(100_000, 100_000),
        guard_matrix(100_000, 100_000),
    ];
    let streamed = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(batches))
        .expect("sparse stream succeeds");
    assert_eq!(streamed.sample_count(), 300_000);
    assert_eq!(streamed.mean().len(), 100_000);
}

/// An explicit in-memory intent also accepts sparse batches.
#[test]
fn in_process_intent_accepts_sparse_batches() {
    let features = zeros_rich_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let collected = SKStandardScaler::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("scaler builds")
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 3, 3])))
        .expect("sparse in-process stream succeeds");
    assert_close_sparse(&collected.mean().to_vec(), &reference.mean().to_vec(), 1e-9);
}

/// The conformance fixture reuses the shared small dense matrix too.
#[test]
fn sparse_stream_matches_a_dense_run_on_the_shared_dataset() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let sparse_stream = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 4])))
        .expect("sparse stream succeeds");
    assert_close_sparse(
        &sparse_stream.mean().to_vec(),
        &reference.mean().to_vec(),
        1e-9,
    );
}
