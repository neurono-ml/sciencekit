//! Chained fitting tests (spec `streaming-pipeline`, task 2.1).

use ndarray::Array2;
use sciencekit_common::SKFeatureBlock;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, SKTestMeanCentering, sk_reopenable_blocks,
    sk_single_block, sk_tiny_blocks,
};

#[test]
fn the_second_stage_is_fitted_on_the_first_stage_output() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    assert_eq!(fitted.number_of_stages(), 2);
    let batch = sk_single_block(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let output = fitted.transform_batch(&batch).unwrap();
    let SKFeatureBlock::Dense(matrix) = output else {
        panic!("the identity finalizer keeps the dense representation");
    };
    // Scaling doubles the data, so the learned means are 12 and 14; the batch
    // [1, 2; 3, 4] scaled to [2, 4; 6, 8] centers to [-10, -10; -6, -6].
    assert_eq!(
        matrix,
        Array2::from_shape_vec((2, 2), vec![-10.0, -10.0, -6.0, -6.0]).unwrap()
    );
}

#[test]
fn a_two_stage_fit_reads_one_pass_per_stage_plus_the_finalizer() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    assert_eq!(fitted.report().number_of_stages(), 2);
    assert_eq!(fitted.report().number_of_batches(), 3);
    assert_eq!(fitted.report().number_of_rows(), 6);
    assert_eq!(fitted.number_of_features(), 2);
}
