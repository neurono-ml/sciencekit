//! Companion tests for `SKIncrementalPCABuilder` and `SKIncrementalPCA`
//! configuration.

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::{SKIncrementalPCA, SKIncrementalPCABuilder};

#[test]
fn incremental_builder_carries_documented_defaults() {
    let estimator = SKIncrementalPCA::<f64>::build().build().unwrap();
    assert_eq!(estimator.number_of_components(), None);
    assert_eq!(estimator.batch_size(), 1024);
    assert!(!estimator.whiten());
    assert_eq!(estimator.maximum_number_of_epochs(), 1);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn incremental_builder_overrides_every_option_and_trait_execution_mode() {
    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .batch_size(128)
        .whiten(true)
        .maximum_number_of_epochs(4)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(estimator.number_of_components(), Some(3));
    assert_eq!(estimator.batch_size(), 128);
    assert!(estimator.whiten());
    assert_eq!(estimator.maximum_number_of_epochs(), 4);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let mut builder = SKIncrementalPCABuilder::<f32>::default();
    SKBuilder::execution_mode(&mut builder, SKExecutionMode::InProcessAsynchronous);
    let f32_estimator = builder.build().unwrap();
    assert_eq!(
        f32_estimator.execution_mode(),
        SKExecutionMode::InProcessAsynchronous
    );
}

#[test]
fn incremental_builder_rejects_zero_components_batch_size_and_epochs() {
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "number_of_components",
            ..
        }
    ));
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .batch_size(0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "batch_size",
            ..
        }
    ));
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .maximum_number_of_epochs(0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "maximum_number_of_epochs",
            ..
        }
    ));
}
