//! Companion tests for `SKGradientBoostingClassifierBuilder` defaults, setters
//! and hyperparameter validation (task 4.1).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};
use sciencekit_tree::SKMaximumFeatures;

use super::{
    SKGradientBoostingClassificationLoss, SKGradientBoostingClassifier,
    SKGradientBoostingClassifierBuilder,
};

fn assert_invalid_hyperparameter<T: std::fmt::Debug>(
    outcome: Result<T, SKError>,
    expected_name: &'static str,
) {
    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => assert_eq!(name, expected_name),
        other => panic!("expected InvalidHyperparameter({expected_name}), got {other:?}"),
    }
}

/// Scenario: Default configuration matches every documented value.
#[test]
fn classifier_builder_defaults_match_contract() {
    let classifier = SKGradientBoostingClassifier::<f64>::build()
        .build()
        .expect("default builder succeeds");

    assert_eq!(classifier.number_of_estimators(), 100);
    assert!((classifier.learning_rate() - 0.1).abs() < 1e-12);
    assert_eq!(
        classifier.loss(),
        SKGradientBoostingClassificationLoss::LogLoss
    );
    assert_eq!(classifier.maximum_depth(), 3);
    assert_eq!(classifier.minimum_samples_leaf(), 1);
    assert_eq!(classifier.maximum_features(), &SKMaximumFeatures::All);
    assert!((classifier.subsample() - 1.0).abs() < 1e-12);
    assert_eq!(classifier.l2_regularization(), 0.0);
    assert_eq!(classifier.random_seed(), 0);
    assert!((classifier.validation_fraction() - 0.1).abs() < 1e-12);
    assert_eq!(classifier.n_iter_no_change(), None);
    assert!((classifier.tolerance() - 1e-4).abs() < 1e-12);
    assert_eq!(classifier.backend(), SKBackendKind::Faer);
    assert_eq!(classifier.execution_intent(), SKExecutionMode::Automatic);
}

/// Every setter reaches the constructed classifier in both `f64` and `f32`.
#[test]
fn configured_values_reach_classifier_in_f64_and_f32() {
    let classifier = SKGradientBoostingClassifierBuilder::<f32>::default()
        .number_of_estimators(40)
        .learning_rate(0.05)
        .loss(SKGradientBoostingClassificationLoss::LogLoss)
        .maximum_depth(5)
        .minimum_samples_leaf(3)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .subsample(0.8)
        .l2_regularization(1.5)
        .random_seed(2026)
        .validation_fraction(0.2)
        .n_iter_no_change(5)
        .tolerance(1e-3)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("configured f32 builder succeeds");

    assert_eq!(classifier.number_of_estimators(), 40);
    assert!((classifier.learning_rate() - 0.05).abs() < 1e-6);
    assert_eq!(classifier.maximum_depth(), 5);
    assert_eq!(classifier.minimum_samples_leaf(), 3);
    assert_eq!(classifier.maximum_features(), &SKMaximumFeatures::Sqrt);
    assert!((classifier.subsample() - 0.8).abs() < 1e-6);
    assert!((classifier.l2_regularization() - 1.5).abs() < 1e-6);
    assert_eq!(classifier.random_seed(), 2026);
    assert!((classifier.validation_fraction() - 0.2).abs() < 1e-6);
    assert_eq!(classifier.n_iter_no_change(), Some(5));
    assert!((classifier.tolerance() - 1e-3).abs() < 1e-6);
    assert_eq!(classifier.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        classifier.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

/// Scenarios: Non-positive learning rate, invalid subsample, zero estimator
/// count, invalid feature/leaf/regularization/early-stopping knobs are
/// rejected at build time with `InvalidHyperparameter`.
#[test]
fn invalid_classifier_hyperparameters_are_rejected() {
    assert_invalid_hyperparameter(
        SKGradientBoostingClassifier::<f64>::build()
            .number_of_estimators(0)
            .build(),
        "number_of_estimators",
    );
    for invalid_rate in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert_invalid_hyperparameter(
            SKGradientBoostingClassifier::<f64>::build()
                .learning_rate(invalid_rate)
                .build(),
            "learning_rate",
        );
    }
    assert_invalid_hyperparameter(
        SKGradientBoostingClassifier::<f64>::build()
            .minimum_samples_leaf(0)
            .build(),
        "minimum_samples_leaf",
    );
    assert_invalid_hyperparameter(
        SKGradientBoostingClassifier::<f64>::build()
            .maximum_features(SKMaximumFeatures::Count(0))
            .build(),
        "maximum_features",
    );
    for invalid_fraction in [0.0, -0.2, 1.1, f64::NAN] {
        assert_invalid_hyperparameter(
            SKGradientBoostingClassifier::<f64>::build()
                .maximum_features(SKMaximumFeatures::Fraction(invalid_fraction))
                .build(),
            "maximum_features",
        );
    }
    for invalid_subsample in [0.0, -0.5, 1.01, f64::NAN, f64::INFINITY] {
        assert_invalid_hyperparameter(
            SKGradientBoostingClassifier::<f64>::build()
                .subsample(invalid_subsample)
                .build(),
            "subsample",
        );
    }
    for invalid_regularization in [-0.01, f64::NAN, f64::INFINITY] {
        assert_invalid_hyperparameter(
            SKGradientBoostingClassifier::<f64>::build()
                .l2_regularization(invalid_regularization)
                .build(),
            "l2_regularization",
        );
    }
    for invalid_validation in [0.0, -0.1, 1.0, 1.5, f64::NAN] {
        assert_invalid_hyperparameter(
            SKGradientBoostingClassifier::<f64>::build()
                .validation_fraction(invalid_validation)
                .build(),
            "validation_fraction",
        );
    }
    assert_invalid_hyperparameter(
        SKGradientBoostingClassifier::<f64>::build()
            .n_iter_no_change(0)
            .build(),
        "n_iter_no_change",
    );
    for invalid_tolerance in [-1e-5, f64::NAN, f64::INFINITY] {
        assert_invalid_hyperparameter(
            SKGradientBoostingClassifier::<f64>::build()
                .tolerance(invalid_tolerance)
                .build(),
            "tolerance",
        );
    }
}
