//! Source and stage error propagation tests (spec `streaming-pipeline`,
//! task 3.4).

use sciencekit_common::{SKError, SKFeatureBlock};
use sprs::TriMat;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{
    SKTestColumnScaling, SKTestIdentityEstimator, SKTestMeanCentering,
    sk_reopenable_blocks, sk_reopenable_failing_blocks, sk_tiny_blocks,
};

#[test]
fn a_mid_stream_source_failure_surfaces_the_central_error() {
    let provider = sk_reopenable_failing_blocks(sk_tiny_blocks::<f64>(), 1);
    let error = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .err().expect("the source fails while the stage reads its first pass");
    assert!(error.to_string().contains("intentional source failure at batch 1"));
}

#[test]
fn a_stage_failure_names_the_stage_index() {
    let provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let fitted = SKStreamingPipeline::start::<f64>()
        .then(SKTestColumnScaling::new(2.0))
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKTestIdentityEstimator)
        .unwrap()
        .fit(provider)
        .unwrap();
    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 1, 3.0);
    let sparse = SKFeatureBlock::Sparse(triplets.to_csr());
    let error = fitted
        .transform_batch(&sparse)
        .err().expect("the mean-centering stage rejects a sparse batch");
    assert!(matches!(error, SKError::Conversion(_)));
    let message = error.to_string();
    assert!(message.contains("stage 1"), "message was {message}");
    assert!(message.contains("sparse"), "message was {message}");
}
