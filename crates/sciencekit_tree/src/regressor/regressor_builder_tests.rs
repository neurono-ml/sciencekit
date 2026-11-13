//! Builder tests for the decision-tree regressor (task 5.1).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFloat};

use crate::criterion::SKTreeCriterion;
use crate::splitter::{SKMaximumFeatures, SKSplitter};

use super::SKDecisionTreeRegressor;

fn assert_invalid<F: SKFloat>(
    builder: Result<SKDecisionTreeRegressor<F>, SKError>,
    expected_name: &str,
) {
    match builder {
        Err(SKError::InvalidHyperparameter { name, .. }) => assert_eq!(name, expected_name),
        _ => panic!("expected InvalidHyperparameter({expected_name})"),
    }
}

/// A tree built without setters carries every documented default.
#[test]
fn default_configuration_matches_the_contract() {
    let tree = SKDecisionTreeRegressor::<f64>::build()
        .build()
        .expect("default configuration builds");
    assert_eq!(tree.criterion(), SKTreeCriterion::SquaredError);
    assert_eq!(tree.maximum_depth(), None);
    assert_eq!(tree.minimum_samples_split(), 2);
    assert_eq!(tree.minimum_samples_leaf(), 1);
    assert_eq!(tree.maximum_features(), &SKMaximumFeatures::All);
    assert_eq!(tree.splitter(), SKSplitter::Best);
    assert_eq!(tree.random_seed(), 0);
    assert_eq!(tree.backend(), SKBackendKind::Faer);
    assert_eq!(tree.execution_intent(), SKExecutionMode::Automatic);
}

/// Every setter is reflected by the built estimator.
#[test]
fn configured_values_reach_the_estimator() {
    let tree = SKDecisionTreeRegressor::<f64>::build()
        .criterion(SKTreeCriterion::AbsoluteError)
        .maximum_depth(3)
        .minimum_samples_split(4)
        .minimum_samples_leaf(2)
        .maximum_features(SKMaximumFeatures::Log2)
        .splitter(SKSplitter::Random)
        .random_seed(5)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("configuration builds");
    assert_eq!(tree.criterion(), SKTreeCriterion::AbsoluteError);
    assert_eq!(tree.maximum_depth(), Some(3));
    assert_eq!(tree.minimum_samples_split(), 4);
    assert_eq!(tree.minimum_samples_leaf(), 2);
    assert_eq!(tree.maximum_features(), &SKMaximumFeatures::Log2);
    assert_eq!(tree.splitter(), SKSplitter::Random);
    assert_eq!(tree.random_seed(), 5);
    assert_eq!(tree.backend(), SKBackendKind::MatrixMultiply);
}

/// A classification criterion is rejected by the regressor.
#[test]
fn classification_criterion_is_rejected() {
    assert_invalid(
        SKDecisionTreeRegressor::<f64>::build()
            .criterion(SKTreeCriterion::Gini)
            .build(),
        "criterion",
    );
}

/// Invalid sample and feature parameters are rejected at build time.
#[test]
fn invalid_parameters_are_rejected() {
    assert_invalid(
        SKDecisionTreeRegressor::<f64>::build()
            .minimum_samples_split(1)
            .build(),
        "minimum_samples_split",
    );
    assert_invalid(
        SKDecisionTreeRegressor::<f64>::build()
            .minimum_samples_leaf(0)
            .build(),
        "minimum_samples_leaf",
    );
    assert_invalid(
        SKDecisionTreeRegressor::<f64>::build()
            .maximum_features(SKMaximumFeatures::Count(0))
            .build(),
        "maximum_features",
    );
    assert_invalid(
        SKDecisionTreeRegressor::<f64>::build()
            .maximum_features(SKMaximumFeatures::Fraction(-0.5))
            .build(),
        "maximum_features",
    );
}
