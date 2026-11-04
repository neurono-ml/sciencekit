//! Float, concurrency and mixed-representation coverage for the sparse
//! streaming standard scaler.

use std::sync::Arc;

use ndarray::{Array2, s};
use sciencekit_common::{SKDataBatch, SKFeatureTransformer, SKUnsupervisedFit};

use super::test_datasets::default_scaler;
use crate::sparse_streaming_test_datasets::{
    OwnedBatchSource, SparseVecSource, assert_close_sparse, chunk_csr, dense_to_csr,
    zeros_rich_features,
};

/// The sparse stream learns the same moments as the dense `fit` on `f32`.
#[test]
fn sparse_stream_matches_dense_fit_on_f32() {
    let features: Array2<f32> = zeros_rich_features().mapv(|value| value as f32);
    let reference = default_scaler().fit(&features).expect("f32 dense fit");
    let streamed = default_scaler()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 3, 3])))
        .expect("f32 sparse stream");
    assert_eq!(streamed.sample_count(), reference.sample_count());
    assert_close_sparse(&streamed.mean().to_vec(), &reference.mean().to_vec(), 1e-5);
    assert_close_sparse(
        &streamed.scale().to_vec(),
        &reference.scale().to_vec(),
        1e-5,
    );
}

/// One fitted sparse-streamed model serves two threads with identical output.
#[test]
fn sparse_streamed_model_serves_concurrent_transforms() {
    let features = zeros_rich_features();
    let model = Arc::new(
        default_scaler()
            .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 3, 3])))
            .expect("sparse stream"),
    );
    let expected = model.transform(&features).expect("transform");

    let handles: Vec<_> = (0..2)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = features.clone();
            std::thread::spawn(move || model.transform(&features).expect("concurrent transform"))
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().expect("worker thread joins"), expected);
    }
}

/// A stream that mixes one dense batch and one sparse batch learns the same
/// moments as the dense `fit` on the concatenated data.
#[test]
fn mixed_dense_and_sparse_stream_matches_the_dense_fit() {
    let features = zeros_rich_features();
    let reference = default_scaler().fit(&features).expect("dense fit");
    let dense_chunk = features.slice(s![0..3, ..]).to_owned();
    let sparse_chunk = dense_to_csr(&features.slice(s![3..8, ..]).to_owned());
    let mut source = OwnedBatchSource::new(vec![
        SKDataBatch::new_dense(dense_chunk, 0, false),
        SKDataBatch::new_sparse(sparse_chunk, 1, true),
    ]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("mixed stream");
    assert_eq!(streamed.sample_count(), reference.sample_count());
    assert_close_sparse(&streamed.mean().to_vec(), &reference.mean().to_vec(), 1e-9);
    assert_close_sparse(
        &streamed.scale().to_vec(),
        &reference.scale().to_vec(),
        1e-9,
    );
}
