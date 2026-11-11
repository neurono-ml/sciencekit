//! Companion tests for random-forest builders (task 2.1).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};
use sciencekit_tree::{SKClassWeight, SKMaximumFeatures, SKTreeCriterion};

use super::{SKMaximumSamples, SKRandomForestClassifier, SKRandomForestRegressor};

fn assert_invalid_hyperparameter<T: std::fmt::Debug>(
    outcome: Result<T, SKError>,
    expected_name: &str,
) {
    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => assert_eq!(name, expected_name),
        other => panic!("expected InvalidHyperparameter({expected_name}), found {other:?}"),
    }
}

/// A classifier built without setters carries every documented default.
#[test]
fn classifier_builder_defaults_match_contract() {
    let forest = SKRandomForestClassifier::<f64>::build()
        .build()
        .expect("default classifier builds");
    assert_eq!(forest.number_of_trees(), 100);
    assert_eq!(forest.criterion(), SKTreeCriterion::Gini);
    assert_eq!(forest.maximum_depth(), None);
    assert_eq!(forest.minimum_samples_split(), 2);
    assert_eq!(forest.minimum_samples_leaf(), 1);
    assert_eq!(forest.maximum_features(), &SKMaximumFeatures::Sqrt);
    assert!(forest.bootstrap());
    assert_eq!(forest.maximum_samples(), &SKMaximumSamples::All);
    assert_eq!(forest.random_seed(), 0);
    assert_eq!(forest.class_weight(), &SKClassWeight::None);
    assert!(!forest.oob_score());
    assert_eq!(forest.backend(), SKBackendKind::Faer);
    assert_eq!(forest.execution_intent(), SKExecutionMode::Automatic);
}

/// A regressor built without setters carries every documented default.
#[test]
fn regressor_builder_defaults_match_contract() {
    let forest = SKRandomForestRegressor::<f64>::build()
        .build()
        .expect("default regressor builds");
    assert_eq!(forest.number_of_trees(), 100);
    assert_eq!(forest.criterion(), SKTreeCriterion::SquaredError);
    assert_eq!(forest.maximum_depth(), None);
    assert_eq!(forest.minimum_samples_split(), 2);
    assert_eq!(forest.minimum_samples_leaf(), 1);
    assert_eq!(forest.maximum_features(), &SKMaximumFeatures::Sqrt);
    assert!(forest.bootstrap());
    assert_eq!(forest.maximum_samples(), &SKMaximumSamples::All);
    assert_eq!(forest.random_seed(), 0);
    assert!(!forest.oob_score());
    assert_eq!(forest.backend(), SKBackendKind::Faer);
    assert_eq!(forest.execution_intent(), SKExecutionMode::Automatic);
}

/// Every setter reaches the built classifier and regressor configurations.
#[test]
fn configured_values_reach_both_estimators() {
    let classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(25)
        .criterion(SKTreeCriterion::Entropy)
        .maximum_depth(6)
        .minimum_samples_split(4)
        .minimum_samples_leaf(2)
        .maximum_features(SKMaximumFeatures::Log2)
        .bootstrap(true)
        .maximum_samples(SKMaximumSamples::Fraction(0.8))
        .random_seed(42)
        .class_weight(SKClassWeight::Balanced)
        .oob_score(true)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("configured classifier builds");
    assert_eq!(classifier.number_of_trees(), 25);
    assert_eq!(classifier.criterion(), SKTreeCriterion::Entropy);
    assert_eq!(classifier.maximum_depth(), Some(6));
    assert_eq!(classifier.minimum_samples_split(), 4);
    assert_eq!(classifier.minimum_samples_leaf(), 2);
    assert_eq!(classifier.maximum_features(), &SKMaximumFeatures::Log2);
    assert_eq!(
        classifier.maximum_samples(),
        &SKMaximumSamples::Fraction(0.8)
    );
    assert_eq!(classifier.random_seed(), 42);
    assert_eq!(classifier.class_weight(), &SKClassWeight::Balanced);
    assert!(classifier.oob_score());
    assert_eq!(classifier.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        classifier.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );

    let regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(15)
        .criterion(SKTreeCriterion::AbsoluteError)
        .maximum_depth(3)
        .minimum_samples_split(5)
        .minimum_samples_leaf(2)
        .maximum_features(SKMaximumFeatures::All)
        .bootstrap(false)
        .random_seed(7)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("configured regressor builds");
    assert_eq!(regressor.number_of_trees(), 15);
    assert_eq!(regressor.criterion(), SKTreeCriterion::AbsoluteError);
    assert!(!regressor.bootstrap());
    assert_eq!(regressor.backend(), SKBackendKind::MatrixMultiply);
}

/// Zero trees are rejected naming `number_of_trees`.
#[test]
fn zero_number_of_trees_is_rejected() {
    assert_invalid_hyperparameter(
        SKRandomForestClassifier::<f64>::build()
            .number_of_trees(0)
            .build(),
        "number_of_trees",
    );
    assert_invalid_hyperparameter(
        SKRandomForestRegressor::<f64>::build()
            .number_of_trees(0)
            .build(),
        "number_of_trees",
    );
}

/// Invalid sample counts and fractions are rejected naming `maximum_samples`.
#[test]
fn invalid_maximum_samples_is_rejected() {
    for invalid in [
        SKMaximumSamples::Count(0),
        SKMaximumSamples::Fraction(0.0),
        SKMaximumSamples::Fraction(-0.5),
        SKMaximumSamples::Fraction(1.2),
        SKMaximumSamples::Fraction(f64::NAN),
    ] {
        assert_invalid_hyperparameter(
            SKRandomForestClassifier::<f64>::build()
                .maximum_samples(invalid.clone())
                .build(),
            "maximum_samples",
        );
        assert_invalid_hyperparameter(
            SKRandomForestRegressor::<f64>::build()
                .maximum_samples(invalid)
                .build(),
            "maximum_samples",
        );
    }
}

/// Requesting OOB scoring while `bootstrap` is false is rejected at build time.
#[test]
fn oob_score_without_bootstrap_is_rejected() {
    assert_invalid_hyperparameter(
        SKRandomForestClassifier::<f64>::build()
            .bootstrap(false)
            .oob_score(true)
            .build(),
        "oob_score",
    );
    assert_invalid_hyperparameter(
        SKRandomForestRegressor::<f64>::build()
            .bootstrap(false)
            .oob_score(true)
            .build(),
        "oob_score",
    );
}

/// Invalid split/leaf/feature/criterion/class-weight values are rejected.
#[test]
fn invalid_tree_constraints_criterion_and_class_weight_are_rejected() {
    assert_invalid_hyperparameter(
        SKRandomForestClassifier::<f64>::build()
            .minimum_samples_split(1)
            .build(),
        "minimum_samples_split",
    );
    assert_invalid_hyperparameter(
        SKRandomForestRegressor::<f64>::build()
            .minimum_samples_leaf(0)
            .build(),
        "minimum_samples_leaf",
    );
    assert_invalid_hyperparameter(
        SKRandomForestClassifier::<f64>::build()
            .maximum_features(SKMaximumFeatures::Fraction(0.0))
            .build(),
        "maximum_features",
    );
    assert_invalid_hyperparameter(
        SKRandomForestClassifier::<f64>::build()
            .criterion(SKTreeCriterion::SquaredError)
            .build(),
        "criterion",
    );
    assert_invalid_hyperparameter(
        SKRandomForestRegressor::<f64>::build()
            .criterion(SKTreeCriterion::Gini)
            .build(),
        "criterion",
    );
    assert_invalid_hyperparameter(
        SKRandomForestClassifier::<f64>::build()
            .class_weight(SKClassWeight::Explicit(vec![0.0, 0.0]))
            .build(),
        "class_weight",
    );
}
