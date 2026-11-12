//! Companion tests for [`SKMiniBatchKMeansBuilder`] defaults, overrides and
//! validation (spec `mini-batch-k-means`, task 4.1).

use sciencekit_common::{SKBackendKind, SKBuilder, SKClusterIdentifier, SKError, SKExecutionMode};

use super::{SKKMeansInitialization, SKMiniBatchKMeans};

fn assert_invalid(error: SKError, expected_name: &'static str) {
    match error {
        SKError::InvalidHyperparameter { name, reason } => {
            assert_eq!(name, expected_name);
            assert!(!reason.is_empty());
        }
        other => panic!("expected InvalidHyperparameter({expected_name}), found {other:?}"),
    }
}

#[test]
fn minibatch_builder_defaults_match_documented_contract() {
    let builder = SKMiniBatchKMeans::<f64>::build();
    let cloned = builder;
    assert!(!format!("{builder:?}").is_empty());
    let estimator = cloned.build().unwrap();

    assert_eq!(estimator.number_of_clusters(), 8);
    assert_eq!(
        estimator.initialization(),
        SKKMeansInitialization::KMeansPlusPlus
    );
    assert_eq!(estimator.number_of_initializations(), 3);
    assert_eq!(estimator.batch_size(), 1024);
    assert_eq!(estimator.initialization_batch_size(), 3072);
    assert_eq!(estimator.maximum_number_of_iterations(), 100);
    assert_eq!(estimator.maximum_number_of_epochs(), 1);
    assert_eq!(estimator.tolerance(), 0.0);
    assert!((estimator.reassignment_ratio() - 0.01).abs() < 1e-12);
    assert_eq!(estimator.random_seed(), 0);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn minibatch_builder_overrides_and_trait_surface_work_on_f32_and_f64() {
    let estimator = SKMiniBatchKMeans::<f32>::build()
        .number_of_clusters(4)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(5)
        .batch_size(128)
        .maximum_number_of_iterations(40)
        .maximum_number_of_epochs(3)
        .tolerance(1e-3)
        .reassignment_ratio(0.05)
        .random_seed(77)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap();

    assert_eq!(estimator.number_of_clusters(), 4);
    assert_eq!(estimator.initialization(), SKKMeansInitialization::Random);
    assert_eq!(estimator.number_of_initializations(), 5);
    assert_eq!(estimator.batch_size(), 128);
    assert_eq!(estimator.initialization_batch_size(), 384);
    assert_eq!(estimator.maximum_number_of_iterations(), 40);
    assert_eq!(estimator.maximum_number_of_epochs(), 3);
    assert!((estimator.tolerance() - 1e-3).abs() < 1e-6);
    assert!((estimator.reassignment_ratio() - 0.05).abs() < 1e-6);
    assert_eq!(estimator.random_seed(), 77);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::OutOfCoreStreaming
    );

    let mut via_trait = SKMiniBatchKMeans::<f64>::build().initialization_batch_size(24);
    SKBuilder::execution_mode(&mut via_trait, SKExecutionMode::InProcessSynchronous);
    let built = SKBuilder::build(via_trait).unwrap();
    assert_eq!(built.initialization_batch_size(), 24);
    assert_eq!(
        built.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn minibatch_builder_rejects_invalid_hyperparameters() {
    assert_invalid(
        SKMiniBatchKMeans::<f64>::build()
            .number_of_clusters(0)
            .build()
            .unwrap_err(),
        "number_of_clusters",
    );
    assert_invalid(
        SKMiniBatchKMeans::<f64>::build()
            .number_of_clusters(SKClusterIdentifier::MAXIMUM_INDEX as usize + 2)
            .build()
            .unwrap_err(),
        "number_of_clusters",
    );
    assert_invalid(
        SKMiniBatchKMeans::<f64>::build()
            .number_of_initializations(0)
            .build()
            .unwrap_err(),
        "number_of_initializations",
    );
    assert_invalid(
        SKMiniBatchKMeans::<f64>::build()
            .batch_size(0)
            .build()
            .unwrap_err(),
        "batch_size",
    );
    assert_invalid(
        SKMiniBatchKMeans::<f64>::build()
            .number_of_clusters(5)
            .initialization_batch_size(4)
            .build()
            .unwrap_err(),
        "initialization_batch_size",
    );
    assert_invalid(
        SKMiniBatchKMeans::<f64>::build()
            .maximum_number_of_iterations(0)
            .build()
            .unwrap_err(),
        "maximum_number_of_iterations",
    );
    assert_invalid(
        SKMiniBatchKMeans::<f64>::build()
            .maximum_number_of_epochs(0)
            .build()
            .unwrap_err(),
        "maximum_number_of_epochs",
    );
    for bad_tolerance in [-0.1_f64, f64::NAN, f64::INFINITY] {
        assert_invalid(
            SKMiniBatchKMeans::<f64>::build()
                .tolerance(bad_tolerance)
                .build()
                .unwrap_err(),
            "tolerance",
        );
    }
    for bad_ratio in [-0.01_f64, 1.01, f64::NAN, f64::INFINITY] {
        assert_invalid(
            SKMiniBatchKMeans::<f64>::build()
                .reassignment_ratio(bad_ratio)
                .build()
                .unwrap_err(),
            "reassignment_ratio",
        );
    }
}
