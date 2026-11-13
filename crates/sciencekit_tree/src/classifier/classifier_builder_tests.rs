//! Builder tests for the decision-tree classifier (task 4.1).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFloat};

use crate::class_weight::SKClassWeight;
use crate::criterion::SKTreeCriterion;
use crate::splitter::{SKMaximumFeatures, SKSplitter};

use super::SKDecisionTreeClassifier;

fn default_estimator<F: SKFloat>() -> SKDecisionTreeClassifier<F> {
    SKDecisionTreeClassifier::<F>::build()
        .build()
        .expect("default configuration builds")
}

/// A tree built without setters carries every documented default.
#[test]
fn default_configuration_matches_the_contract() {
    let tree = default_estimator::<f64>();
    assert_eq!(tree.criterion(), SKTreeCriterion::Gini);
    assert_eq!(tree.maximum_depth(), None);
    assert_eq!(tree.minimum_samples_split(), 2);
    assert_eq!(tree.minimum_samples_leaf(), 1);
    assert_eq!(tree.maximum_features(), &SKMaximumFeatures::All);
    assert_eq!(tree.splitter(), SKSplitter::Best);
    assert_eq!(tree.random_seed(), 0);
    assert_eq!(tree.class_weight(), &SKClassWeight::None);
    assert_eq!(tree.backend(), SKBackendKind::Faer);
    assert_eq!(tree.execution_intent(), SKExecutionMode::Automatic);
}

/// Every setter is reflected by the built estimator.
#[test]
fn configured_values_reach_the_estimator() {
    let tree = SKDecisionTreeClassifier::<f64>::build()
        .criterion(SKTreeCriterion::Entropy)
        .maximum_depth(4)
        .minimum_samples_split(6)
        .minimum_samples_leaf(3)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .splitter(SKSplitter::Random)
        .random_seed(99)
        .class_weight(SKClassWeight::Balanced)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("configuration builds");
    assert_eq!(tree.criterion(), SKTreeCriterion::Entropy);
    assert_eq!(tree.maximum_depth(), Some(4));
    assert_eq!(tree.minimum_samples_split(), 6);
    assert_eq!(tree.minimum_samples_leaf(), 3);
    assert_eq!(tree.maximum_features(), &SKMaximumFeatures::Sqrt);
    assert_eq!(tree.splitter(), SKSplitter::Random);
    assert_eq!(tree.random_seed(), 99);
    assert_eq!(tree.class_weight(), &SKClassWeight::Balanced);
    assert_eq!(tree.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        tree.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

fn assert_invalid<F: SKFloat>(
    builder: Result<SKDecisionTreeClassifier<F>, SKError>,
    expected_name: &str,
) {
    match builder {
        Err(SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, expected_name)
        }
        _ => panic!("expected InvalidHyperparameter({expected_name})"),
    }
}

/// Invalid depth, split and leaf parameters are rejected at build time.
#[test]
fn invalid_sample_parameters_are_rejected() {
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .minimum_samples_split(1)
            .build(),
        "minimum_samples_split",
    );
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .minimum_samples_leaf(0)
            .build(),
        "minimum_samples_leaf",
    );
}

/// A criterion from the other estimator family is rejected.
#[test]
fn regression_criterion_is_rejected() {
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .criterion(SKTreeCriterion::SquaredError)
            .build(),
        "criterion",
    );
}

/// Invalid feature-subsampling configurations are rejected.
#[test]
fn invalid_feature_subsampling_is_rejected() {
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_features(SKMaximumFeatures::Count(0))
            .build(),
        "maximum_features",
    );
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_features(SKMaximumFeatures::Fraction(0.0))
            .build(),
        "maximum_features",
    );
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_features(SKMaximumFeatures::Fraction(1.5))
            .build(),
        "maximum_features",
    );
}

/// An explicit class-weight vector must be non-empty, non-negative and not all
/// zero.
#[test]
fn invalid_class_weights_are_rejected() {
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .class_weight(SKClassWeight::Explicit(vec![]))
            .build(),
        "class_weight",
    );
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .class_weight(SKClassWeight::Explicit(vec![1.0, -1.0]))
            .build(),
        "class_weight",
    );
    assert_invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .class_weight(SKClassWeight::Explicit(vec![0.0, 0.0]))
            .build(),
        "class_weight",
    );
}
