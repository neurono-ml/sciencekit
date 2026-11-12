//! Streaming pipeline acceptance coverage (spec `streaming-pipeline`).

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

#[test]
fn a_tiny_stream_fits_and_transforms_dense_batches() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    let output = fitted
        .transform_batch(&sk_single_block(2, 2, vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    assert_eq!(output.number_of_rows(), 2);
    assert_eq!(fitted.report().number_of_rows(), 6);
}

#[test]
fn a_large_reopenable_stream_never_materializes_the_source() {
    let provider = sk_reopenable_blocks(generated_blocks(500));
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .epochs(2)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    assert_eq!(fitted.report().number_of_rows(), 1_000);
    assert_eq!(fitted.report().materialized_bytes(), 0);
    assert_eq!(fitted.report().number_of_epochs(), 2);
}

#[test]
fn a_one_shot_stream_fits_within_the_budget() {
    use crate::streaming::testing::SKTestBlocksSource;

    let mut source = SKTestBlocksSource::new(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(&mut source)
        .unwrap();
    assert!(fitted.report().materialized_bytes() > 0);
}
