//! Construction and validation tests (spec `streaming-pipeline`, task 1.1).

use sciencekit_common::SKBuilder;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::streaming::SKStreamingPipeline;
use crate::streaming::testing::{SKTestMeanCentering, sk_tiny_blocks};

#[test]
fn an_empty_streaming_pipeline_is_rejected() {
    let result = SKStreamingPipeline::start::<f64>()
        .finalize_estimator(SKStandardScaler::build().build().unwrap());
    match result {
        Err(sciencekit_common::SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "stages");
        }
        _ => panic!("an empty streaming pipeline must be rejected"),
    }
}

#[test]
fn zero_epochs_are_rejected() {
    let result = SKStreamingPipeline::start::<f64>()
        .then(SKTestMeanCentering::new())
        .epochs(0)
        .finalize_estimator(SKStandardScaler::build().build().unwrap());
    match result {
        Err(sciencekit_common::SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "epochs");
        }
        _ => panic!("zero epochs must be rejected"),
    }
}

#[test]
fn a_zero_memory_budget_is_rejected() {
    let result = SKStreamingPipeline::start::<f64>()
        .then(SKTestMeanCentering::new())
        .memory_budget_bytes(0)
        .finalize_estimator(SKStandardScaler::build().build().unwrap());
    match result {
        Err(sciencekit_common::SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "memory_budget_bytes");
        }
        _ => panic!("a zero memory budget must be rejected"),
    }
}

#[test]
fn a_valid_pipeline_is_configured_with_the_default_backend() {
    use sciencekit_common::SKBackendKind;

    let provider = super::testing::sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let configured = SKStreamingPipeline::start::<f64>()
        .then(SKTestMeanCentering::new())
        .finalize_estimator(SKStandardScaler::build().build().unwrap())
        .unwrap();
    let fitted = configured.fit(provider).unwrap();
    assert_eq!(fitted.backend(), SKBackendKind::Faer);
    assert_eq!(fitted.number_of_stages(), 1);
    assert_eq!(fitted.report().number_of_epochs(), 1);
    assert_eq!(fitted.report().number_of_batches(), 3);
    assert_eq!(fitted.report().number_of_rows(), 6);
}
