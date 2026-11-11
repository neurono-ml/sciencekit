//! Companion tests for [`SKBaggingClassifierBuilder`] and
//! [`SKBaggingRegressorBuilder`] defaults and hyperparameter validation.

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};

use crate::{SKBaggingClassifier, SKBaggingRegressor, SKMaximumSamples};

#[test]
fn classifier_and_regressor_builders_match_documented_defaults() {
    let base_classifier = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    let classifier = SKBaggingClassifier::build(base_classifier).build().unwrap();
    assert_eq!(classifier.number_of_estimators(), 10);
    assert_eq!(classifier.maximum_samples(), &SKMaximumSamples::All);
    assert_eq!(classifier.maximum_features(), &SKMaximumFeatures::All);
    assert!(classifier.bootstrap());
    assert!(!classifier.bootstrap_features());
    assert!(!classifier.oob_score());
    assert_eq!(classifier.random_seed(), 0);
    assert_eq!(classifier.backend(), SKBackendKind::Faer);
    assert_eq!(classifier.execution_intent(), SKExecutionMode::Automatic);

    let base_regressor = SKDecisionTreeRegressor::<f64>::build().build().unwrap();
    let regressor = SKBaggingRegressor::build(base_regressor).build().unwrap();
    assert_eq!(regressor.number_of_estimators(), 10);
    assert_eq!(regressor.maximum_samples(), &SKMaximumSamples::All);
    assert_eq!(regressor.maximum_features(), &SKMaximumFeatures::All);
    assert!(regressor.bootstrap());
    assert!(!regressor.bootstrap_features());
    assert!(!regressor.oob_score());
    assert_eq!(regressor.random_seed(), 0);
    assert_eq!(regressor.backend(), SKBackendKind::Faer);
    assert_eq!(regressor.execution_intent(), SKExecutionMode::Automatic);
}

#[test]
fn explicit_builder_setters_transfer_every_hyperparameter() {
    let base_classifier = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(3)
        .build()
        .unwrap();
    let classifier = SKBaggingClassifier::build(base_classifier)
        .number_of_estimators(24)
        .maximum_samples(SKMaximumSamples::Fraction(0.75))
        .maximum_features(SKMaximumFeatures::Count(4))
        .bootstrap(true)
        .bootstrap_features(true)
        .oob_score(true)
        .random_seed(99)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert_eq!(classifier.base_estimator().maximum_depth(), Some(3));
    assert_eq!(classifier.number_of_estimators(), 24);
    assert_eq!(
        classifier.maximum_samples(),
        &SKMaximumSamples::Fraction(0.75)
    );
    assert_eq!(classifier.maximum_features(), &SKMaximumFeatures::Count(4));
    assert!(classifier.bootstrap());
    assert!(classifier.bootstrap_features());
    assert!(classifier.oob_score());
    assert_eq!(classifier.random_seed(), 99);
    assert_eq!(classifier.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        classifier.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn zero_estimator_count_is_rejected_naming_the_field() {
    let base_classifier = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    let classifier_error = SKBaggingClassifier::build(base_classifier)
        .number_of_estimators(0)
        .build()
        .unwrap_err();
    assert!(matches!(
        classifier_error,
        SKError::InvalidHyperparameter {
            name: "number_of_estimators",
            ..
        }
    ));

    let base_regressor = SKDecisionTreeRegressor::<f64>::build().build().unwrap();
    let regressor_error = SKBaggingRegressor::build(base_regressor)
        .number_of_estimators(0)
        .build()
        .unwrap_err();
    assert!(matches!(
        regressor_error,
        SKError::InvalidHyperparameter {
            name: "number_of_estimators",
            ..
        }
    ));
}

#[test]
fn invalid_sample_and_feature_counts_or_fractions_are_rejected() {
    let invalid_samples = [
        SKMaximumSamples::Count(0),
        SKMaximumSamples::Fraction(0.0),
        SKMaximumSamples::Fraction(-0.2),
        SKMaximumSamples::Fraction(1.25),
        SKMaximumSamples::Fraction(f64::NAN),
        SKMaximumSamples::Fraction(f64::INFINITY),
    ];
    invalid_samples.into_iter().for_each(|sample_policy| {
        let base = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
        let error = SKBaggingClassifier::build(base)
            .maximum_samples(sample_policy)
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "maximum_samples",
                ..
            }
        ));
    });

    let invalid_features = [
        SKMaximumFeatures::Count(0),
        SKMaximumFeatures::Fraction(0.0),
        SKMaximumFeatures::Fraction(-0.1),
        SKMaximumFeatures::Fraction(1.1),
        SKMaximumFeatures::Fraction(f64::NAN),
    ];
    invalid_features.into_iter().for_each(|feature_policy| {
        let base = SKDecisionTreeRegressor::<f64>::build().build().unwrap();
        let error = SKBaggingRegressor::build(base)
            .maximum_features(feature_policy)
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "maximum_features",
                ..
            }
        ));
    });
}

#[test]
fn oob_score_without_bootstrap_is_rejected_at_build() {
    let base = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    let error = SKBaggingClassifier::build(base)
        .bootstrap(false)
        .oob_score(true)
        .build()
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "oob_score",
            ..
        }
    ));
}
