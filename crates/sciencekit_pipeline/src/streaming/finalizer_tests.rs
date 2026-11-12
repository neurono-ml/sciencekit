//! Bounded batch-finalizer tests (spec `streaming-pipeline`, task 3.3).

use sciencekit_common::SKError;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, sk_reopenable_blocks, sk_tiny_blocks,
};

fn configured_pipeline(
    memory_budget_bytes: usize,
) -> crate::streaming::SKStreamingPipelineConfigured<f64, SKTestIdentityEstimator> {
    SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .memory_budget_bytes(memory_budget_bytes)
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
}

#[test]
fn the_finalizer_fits_within_a_sufficient_budget() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    // Three 2×2 f64 blocks fit exactly in 96 bytes.
    let fitted = configured_pipeline(96).fit(provider).unwrap();
    assert_eq!(fitted.report().finalizer_bytes(), 96);
    assert_eq!(fitted.number_of_features(), 2);
}

#[test]
fn the_finalizer_rejects_an_accumulation_over_the_budget() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let error = configured_pipeline(95)
        .fit(provider)
        .err().expect("the third block would exceed the 95-byte budget");
    match error {
        SKError::BatchBufferOversize {
            batch_bytes,
            required_bytes,
            available_bytes,
        } => {
            assert_eq!(batch_bytes, 32);
            assert_eq!(required_bytes, 96);
            assert_eq!(available_bytes, 95);
        }
        other => panic!("expected a batch-buffer oversize error, got {other:?}"),
    }
}
