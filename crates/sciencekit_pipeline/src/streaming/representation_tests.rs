//! Dense/sparse flow and no-densification tests (spec `streaming-pipeline`,
//! task 3.2).

use sciencekit_common::SKFeatureBlock;
use sprs::TriMat;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, sk_reopenable_blocks, sk_single_block,
    sk_tiny_blocks,
};

fn sparse_batch() -> SKFeatureBlock<f64> {
    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 1, 3.0);
    triplets.add_triplet(1, 0, 5.0);
    SKFeatureBlock::Sparse(triplets.to_csr())
}

#[test]
fn a_sparse_batch_stays_csr_and_is_not_densified() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    let batch = sparse_batch();
    let output = fitted.transform_batch(&batch).unwrap();
    let SKFeatureBlock::Sparse(matrix) = output else {
        panic!("a sparse batch must stay CSR through sparse-capable stages");
    };
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 2);
    assert_eq!(matrix.nnz(), 2);
    assert!(matrix.data().iter().any(|value| *value == 6.0));
    assert!(matrix.data().iter().any(|value| *value == 10.0));
}

#[test]
fn a_dense_batch_stays_dense() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    let output = fitted
        .transform_batch(&sk_single_block(1, 2, vec![1.0, 2.0]))
        .unwrap();
    assert!(!output.is_sparse());
}
