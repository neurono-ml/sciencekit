//! Sparse streaming conformance tests for `SKMinMaxScaler`.
//!
//! Sparse extrema include the implicit zeros of every column, unless a column
//! is explicitly dense in every batch. The learned minimum, range and scale
//! must equal the dense stream and the dense `fit` on the concatenated data.

use sciencekit_common::{SKBuilder, SKExecutionMode, SKUnsupervisedFit};

use super::core_implementation::SKMinMaxScaler;
use super::test_datasets::{chunked_source, default_scaler};
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
    for (actual, expected) in [
        (sparse_stream.minimum(), reference.minimum()),
        (sparse_stream.range(), reference.range()),
        (sparse_stream.scale(), reference.scale()),
    ] {
        assert_close_sparse(&actual.to_vec(), &expected.to_vec(), 1e-9);
    }
    assert_close_sparse(
        &sparse_stream.minimum().to_vec(),
        &dense_stream.minimum().to_vec(),
        1e-9,
    );
}

/// Implicit zeros extend a column's minimum and maximum down to zero.
#[test]
fn implicit_zeros_participate_in_the_extrema() {
    let features = ndarray::array![[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]];
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let sparse_stream = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[1, 2])))
        .expect("sparse stream succeeds");
    assert_close_sparse(
        &sparse_stream.minimum().to_vec(),
        &reference.minimum().to_vec(),
        1e-12,
    );
    assert_close_sparse(
        &sparse_stream.range().to_vec(),
        &reference.range().to_vec(),
        1e-12,
    );
    assert_eq!(sparse_stream.minimum()[0], 0.0);
    assert_eq!(sparse_stream.minimum()[1], 2.0);
}

/// A column explicitly dense in every batch keeps its true minimum: zeros are
/// not assumed just because the representation is sparse.
#[test]
fn explicitly_dense_columns_ignore_the_implicit_zeros() {
    let features = ndarray::array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let sparse_stream = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[1, 2])))
        .expect("sparse stream succeeds");
    assert_close_sparse(
        &sparse_stream.minimum().to_vec(),
        &reference.minimum().to_vec(),
        1e-12,
    );
    assert_eq!(sparse_stream.minimum()[0], 1.0);
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
    assert_eq!(streamed.minimum().len(), 100_000);
}

/// An explicit in-memory intent also accepts sparse batches.
#[test]
fn in_process_intent_accepts_sparse_batches() {
    let features = zeros_rich_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let collected = SKMinMaxScaler::<f64>::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("scaler builds")
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 3, 3])))
        .expect("sparse in-process stream succeeds");
    assert_close_sparse(
        &collected.minimum().to_vec(),
        &reference.minimum().to_vec(),
        1e-9,
    );
}
