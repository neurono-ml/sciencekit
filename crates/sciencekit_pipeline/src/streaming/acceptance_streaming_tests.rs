//! Acceptance coverage for the streaming pipeline (spec `streaming-pipeline`,
//! task 4.1).
//!
//! Covers tiny and large streams, one and many epochs, dense and sparse flow,
//! chained transformers, the finalizer budget and metric production.

use sciencekit_common::SKFeatureBlock;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, SKTestMeanCentering, sk_reopenable_blocks,
    sk_single_block, sk_tiny_blocks,
};

fn generated_blocks(number_of_blocks: usize) -> Vec<SKFeatureBlock<f64>> {
    (0..number_of_blocks)
        .map(|index| {
            let base = index as f64;
            sk_single_block(2, 2, vec![base, base + 1.0, base + 2.0, base + 3.0])
        })
        .collect()
}

fn fitted_chain(blocks: Vec<SKFeatureBlock<f64>>, epochs: usize) -> crate::streaming::SKFittedStreamingPipeline<f64, crate::streaming::testing::SKTestIdentityModel> {
    let provider = sk_reopenable_blocks(blocks);
    SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .epochs(epochs)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap()
}

#[test]
fn a_tiny_stream_produces_a_centered_metric_near_zero() {
    let fitted = fitted_chain(sk_tiny_blocks::<f64>(), 1);
    let transformed = fitted
        .transform_batch(&sk_single_block(6, 2, (1..=12).map(f64::from).collect()))
        .unwrap();
    let SKFeatureBlock::Dense(matrix) = transformed else {
        panic!("the identity finalizer keeps the dense representation");
    };
    // Metric production: every column of an exactly centered stream has mean 0.
    let means = matrix.sum_axis(ndarray::Axis(0)).mapv(|value| value / 6.0);
    assert!(means.iter().all(|value| value.abs() < 1e-9));
}

#[test]
fn a_large_stream_fits_without_materializing_the_source() {
    let fitted = fitted_chain(generated_blocks(1_000), 2);
    assert_eq!(fitted.report().number_of_rows(), 2_000);
    assert_eq!(fitted.report().materialized_bytes(), 0);
    assert_eq!(fitted.report().number_of_epochs(), 2);
    let output = fitted
        .transform_batch(&sk_single_block(2, 2, vec![0.0, 1.0, 2.0, 3.0]))
        .unwrap();
    assert_eq!(output.number_of_rows(), 2);
}

#[test]
fn a_sparse_batch_flows_through_sparse_capable_stages() {
    use sprs::TriMat;

    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    let mut triplets = TriMat::new((1, 2));
    triplets.add_triplet(0, 0, 4.0);
    let output = fitted
        .transform_batch(&SKFeatureBlock::Sparse(triplets.to_csr()))
        .unwrap();
    assert!(output.is_sparse());
}

#[test]
fn the_finalizer_budget_is_enforced_on_a_large_stream() {
    let provider = sk_reopenable_blocks(generated_blocks(10));
    let error = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .memory_budget_bytes(64)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .err().expect("ten dense blocks do not fit a 64-byte budget");
    assert!(matches!(
        error,
        sciencekit_common::SKError::BatchBufferOversize { .. }
    ));
}
