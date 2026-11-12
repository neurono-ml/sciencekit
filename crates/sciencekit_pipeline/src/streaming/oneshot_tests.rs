//! One-shot provider tests (spec `streaming-pipeline`, task 2.2).

use sciencekit_common::SKError;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{
    SKTestBlocksSource, SKTestColumnScaling, SKTestIdentityEstimator, SKTestMeanCentering,
    sk_tiny_blocks,
};

#[test]
fn a_one_shot_provider_fits_a_single_stage() {
    let mut source = SKTestBlocksSource::new(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(&mut source)
        .unwrap();
    assert_eq!(fitted.report().number_of_rows(), 6);
    assert!(fitted.report().materialized_bytes() > 0);
}

#[test]
fn a_one_shot_provider_materializes_within_the_budget_for_a_chain() {
    let mut source = SKTestBlocksSource::new(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(&mut source)
        .unwrap();
    assert_eq!(fitted.number_of_stages(), 2);
    // Three dense 2×2 f64 blocks: 3 × 4 × 8 = 96 resident bytes.
    assert_eq!(fitted.report().materialized_bytes(), 96);
}

#[test]
fn a_one_shot_provider_rejects_multiple_epochs() {
    let mut source = SKTestBlocksSource::new(sk_tiny_blocks::<f64>());
    let error = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .epochs(2)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(&mut source)
        .err()
        .expect("a one-shot provider cannot replay two epochs");
    assert!(matches!(
        error,
        SKError::UnsupportedOperation { operation, .. } if operation == "streaming_pipeline_fit"
    ));
}

#[test]
fn a_one_shot_provider_over_budget_returns_a_structured_error() {
    let mut source = SKTestBlocksSource::new(sk_tiny_blocks::<f64>());
    let error = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .memory_budget_bytes(32)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(&mut source)
        .err()
        .expect("96 resident bytes exceed a 32-byte budget");
    assert!(matches!(error, SKError::BatchBufferOversize { .. }));
}
