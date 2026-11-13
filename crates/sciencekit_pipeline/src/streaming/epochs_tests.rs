//! Multi-epoch fitting and epoch-reporting tests (spec `streaming-pipeline`,
//! task 2.3).

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, sk_reopenable_blocks, sk_tiny_blocks,
};

#[test]
fn a_reopenable_provider_replays_every_configured_epoch() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .epochs(3)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    assert_eq!(fitted.report().number_of_epochs(), 3);
}

#[test]
fn the_epoch_count_does_not_multiply_the_finalizer_rows() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .epochs(4)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    // The finalizer takes one accumulation pass regardless of stage epochs.
    assert_eq!(fitted.report().number_of_batches(), 3);
    assert_eq!(fitted.report().number_of_rows(), 6);
    assert_eq!(fitted.report().number_of_epochs(), 4);
}
