//! Float and mixed-representation coverage for the sparse streaming OLS fit.

use ndarray::s;
use sciencekit_common::{SKSupervisedBatch, SKSupervisedFit};

use super::test_datasets::{default_regressor, small_features, small_targets};
use crate::sparse_streaming_test_datasets::{
    OwnedSupervisedBatchSource, SparseSupervisedSource, assert_close_sparse, chunk_pairs_csr,
    dense_to_csr,
};

/// The sparse stream accumulates the same Gram state as the dense `fit` on
/// `f32`.
#[test]
fn sparse_stream_matches_dense_fit_on_f32() {
    let features_f64 = small_features();
    let targets_f64 = small_targets(&features_f64);
    let features = features_f64.mapv(|value| value as f32);
    let targets = targets_f64.mapv(|value| value as f32);

    let reference = default_regressor()
        .fit(features.view(), targets.view())
        .expect("f32 dense fit");
    let streamed = default_regressor()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[1, 1, 2],
        )))
        .expect("f32 sparse stream");

    assert_eq!(streamed.row_count(), reference.row_count());
    assert_close_sparse(
        streamed.coefficients().as_slice().expect("contiguous"),
        reference.coefficients().as_slice().expect("contiguous"),
        1e-4,
    );
    assert_close_sparse(&[streamed.intercept()], &[reference.intercept()], 1e-4);
}

/// A stream that mixes one dense batch and one sparse batch accumulates the
/// same Gram state as the dense `fit` on the concatenated data.
#[test]
fn mixed_dense_and_sparse_stream_matches_the_dense_fit() {
    let features = small_features();
    let targets = small_targets(&features);
    let reference = default_regressor()
        .fit(features.view(), targets.view())
        .expect("dense fit");

    let dense_chunk = features.slice(s![0..2, ..]).to_owned();
    let dense_targets = targets.slice(s![0..2]).to_owned();
    let sparse_chunk = dense_to_csr(&features.slice(s![2..4, ..]).to_owned());
    let sparse_targets = targets.slice(s![2..4]).to_owned();
    let mut source = OwnedSupervisedBatchSource::new(vec![
        SKSupervisedBatch::new_dense(dense_chunk, dense_targets, 0, false)
            .expect("dense batch pairs"),
        SKSupervisedBatch::new_sparse(sparse_chunk, sparse_targets, 1, true)
            .expect("sparse batch pairs"),
    ]);
    let streamed = default_regressor()
        .fit_streaming(&mut source)
        .expect("mixed stream");

    assert_eq!(streamed.row_count(), reference.row_count());
    assert_close_sparse(
        streamed.coefficients().as_slice().expect("contiguous"),
        reference.coefficients().as_slice().expect("contiguous"),
        1e-9,
    );
    assert_close_sparse(&[streamed.intercept()], &[reference.intercept()], 1e-9);
}
