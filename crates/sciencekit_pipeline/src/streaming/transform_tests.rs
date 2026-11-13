//! Per-batch transform tests (spec `streaming-pipeline`, task 3.1).
//!
//! The streaming pipeline is compared against the equivalent batch path: the
//! same transformers fitted directly on the same data and applied in order.

use sciencekit_common::SKFeatureBlock;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::SKStreamingTransformer;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, SKTestMeanCentering, sk_reopenable_blocks,
    sk_single_block, sk_tiny_blocks,
};

fn streaming_output(batch: &SKFeatureBlock<f64>) -> SKFeatureBlock<f64> {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap()
        .transform_batch(batch)
        .unwrap()
}

fn batch_output(batch: &SKFeatureBlock<f64>) -> SKFeatureBlock<f64> {
    let scaling = SKTestColumnScaling::new(2.0);
    let centering = SKTestMeanCentering::new();
    let mut raw = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let scaling_model = scaling.fit_streaming(&mut raw, 1).unwrap();
    let scaled_blocks: Vec<SKFeatureBlock<f64>> = sk_tiny_blocks::<f64>()
        .iter()
        .map(|block| scaling.transform_batch(&scaling_model, block).unwrap())
        .collect();
    let mut scaled = sk_reopenable_blocks(scaled_blocks);
    let centering_model = centering.fit_streaming(&mut scaled, 1).unwrap();
    let scaled_batch = scaling.transform_batch(&scaling_model, batch).unwrap();
    centering
        .transform_batch(&centering_model, &scaled_batch)
        .unwrap()
}

#[test]
fn streaming_transform_equals_the_equivalent_batch_transform() {
    let batch = sk_single_block(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let streaming = streaming_output(&batch);
    let batch_output = batch_output(&batch);
    let (SKFeatureBlock::Dense(streaming), SKFeatureBlock::Dense(batch_output)) =
        (streaming, batch_output)
    else {
        panic!("the identity finalizer keeps the dense representation");
    };
    assert_eq!(streaming, batch_output);
}

#[test]
fn streaming_transform_matches_on_unseen_rows() {
    let batch = sk_single_block(3, 2, vec![100.0, -50.0, 0.0, 0.0, 7.5, 12.5]);
    let streaming = streaming_output(&batch);
    let batch_output = batch_output(&batch);
    let (SKFeatureBlock::Dense(streaming), SKFeatureBlock::Dense(batch_output)) =
        (streaming, batch_output)
    else {
        panic!("the identity finalizer keeps the dense representation");
    };
    assert_eq!(streaming, batch_output);
}
