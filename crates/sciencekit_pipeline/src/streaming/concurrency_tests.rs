//! Repeated-fit determinism and concurrent-transform tests (spec
//! `streaming-pipeline`, task 3.5).

use sciencekit_common::SKFeatureBlock;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::SKFittedStreamingPipeline;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, SKTestIdentityModel, SKTestMeanCentering,
    sk_reopenable_blocks, sk_single_block, sk_tiny_blocks,
};

type FittedPipeline = SKFittedStreamingPipeline<f64, SKTestIdentityModel>;

fn build() -> FittedPipeline {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap()
}

#[test]
fn fitting_the_same_stream_twice_produces_identical_output() {
    let batch = sk_single_block(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let first = build().transform_batch(&batch).unwrap();
    let second = build().transform_batch(&batch).unwrap();
    let (SKFeatureBlock::Dense(first), SKFeatureBlock::Dense(second)) = (first, second) else {
        panic!("the identity finalizer keeps the dense representation");
    };
    assert_eq!(first, second);
}

#[test]
fn the_fitted_pipeline_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FittedPipeline>();
}

#[test]
fn concurrent_transform_agrees_with_the_sequential_result() {
    let fitted = build();
    let batch = sk_single_block(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
    let expected = fitted.transform_batch(&batch).unwrap();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let pipeline = &fitted;
            let batch = &batch;
            handles.push(scope.spawn(move || pipeline.transform_batch(batch).unwrap()));
        }
        for handle in handles {
            let result = handle.join().unwrap();
            let (SKFeatureBlock::Dense(result), SKFeatureBlock::Dense(expected)) =
                (result, expected.clone())
            else {
                panic!("the identity finalizer keeps the dense representation");
            };
            assert_eq!(result, expected);
        }
    });
}
