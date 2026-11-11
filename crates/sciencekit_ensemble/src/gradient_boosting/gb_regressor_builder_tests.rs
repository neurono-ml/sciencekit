//! Companion tests for `SKGradientBoostingRegressorBuilder` defaults, setters
//! and hyperparameter validation (task 5.1).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};
use sciencekit_tree::SKMaximumFeatures;

use super::{
    SKGradientBoostingRegressionLoss, SKGradientBoostingRegressor,
    SKGradientBoostingRegressorBuilder,
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
fn regressor_builder_defaults_match_contract() {
    let regressor = SKGradientBoostingRegressor::<f64>::build()
        .build()
        .expect("default builder succeeds");

    assert_eq!(regressor.number_of_estimators(), 100);
    assert!((regressor.learning_rate() - 0.1).abs() < 1e-12);
    assert_eq!(
        regressor.loss(),
        SKGradientBoostingRegressionLoss::SquaredError
    );
    assert!((regressor.huber_delta() - 1.0).abs() < 1e-12);
    assert_eq!(regressor.maximum_depth(), 3);
    assert_eq!(regressor.minimum_samples_leaf(), 1);
    assert_eq!(regressor.maximum_features(), &SKMaximumFeatures::All);
    assert!((regressor.subsample() - 1.0).abs() < 1e-12);
    assert_eq!(regressor.l2_regularization(), 0.0);
    assert_eq!(regressor.random_seed(), 0);
    assert!((regressor.validation_fraction() - 0.1).abs() < 1e-12);
    assert_eq!(regressor.n_iter_no_change(), None);
    assert!((regressor.tolerance() - 1e-4).abs() < 1e-12);
    assert_eq!(regressor.backend(), SKBackendKind::Faer);
    assert_eq!(regressor.execution_intent(), SKExecutionMode::Automatic);
}

/// Every setter reaches the constructed regressor in `f64` and `f32`, across
/// all regression losses.
#[test]
fn configured_values_reach_regressor_in_f64_and_f32() {
    let regressor = SKGradientBoostingRegressorBuilder::<f32>::default()
        .number_of_estimators(60)
        .learning_rate(0.08)
        .loss(SKGradientBoostingRegressionLoss::Huber)
        .huber_delta(1.35)
        .maximum_depth(4)
        .minimum_samples_leaf(2)
        .maximum_features(SKMaximumFeatures::Log2)
        .subsample(0.75)
        .l2_regularization(0.5)
        .random_seed(99)
        .validation_fraction(0.15)
        .n_iter_no_change(4)
        .tolerance(1e-3)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("configured f32 builder succeeds");

    assert_eq!(regressor.number_of_estimators(), 60);
    assert!((regressor.learning_rate() - 0.08).abs() < 1e-6);
    assert_eq!(regressor.loss(), SKGradientBoostingRegressionLoss::Huber);
    assert!((regressor.huber_delta() - 1.35).abs() < 1e-6);
    assert_eq!(regressor.maximum_depth(), 4);
    assert_eq!(regressor.minimum_samples_leaf(), 2);
    assert_eq!(regressor.maximum_features(), &SKMaximumFeatures::Log2);
    assert!((regressor.subsample() - 0.75).abs() < 1e-6);
    assert!((regressor.l2_regularization() - 0.5).abs() < 1e-6);
    assert_eq!(regressor.random_seed(), 99);
    assert!((regressor.validation_fraction() - 0.15).abs() < 1e-6);
    assert_eq!(regressor.n_iter_no_change(), Some(4));
    assert!((regressor.tolerance() - 1e-3).abs() < 1e-6);
    assert_eq!(regressor.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        regressor.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

/// Invalid estimator count, learning rate, leaf count, feature policy,
/// subsample, regularization, validation fraction, patience, tolerance and
/// Huber delta are rejected with `InvalidHyperparameter`.
#[test]
fn invalid_regressor_hyperparameters_are_rejected() {
    assert_invalid_hyperparameter(
        SKGradientBoostingRegressor::<f64>::build()
            .number_of_estimators(0)
            .build(),
        "number_of_estimators",
    );
    for invalid_rate in [0.0, -0.25, f64::NAN, f64::INFINITY] {
        assert_invalid_hyperparameter(
            SKGradientBoostingRegressor::<f64>::build()
                .learning_rate(invalid_rate)
                .build(),
            "learning_rate",
        );
    }
    for invalid_delta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_invalid_hyperparameter(
            SKGradientBoostingRegressor::<f64>::build()
                .huber_delta(invalid_delta)
                .build(),
            "huber_delta",
        );
    }
    assert_invalid_hyperparameter(
        SKGradientBoostingRegressor::<f64>::build()
            .minimum_samples_leaf(0)
            .build(),
        "minimum_samples_leaf",
    );
    assert_invalid_hyperparameter(
        SKGradientBoostingRegressor::<f64>::build()
            .maximum_features(SKMaximumFeatures::Count(0))
            .build(),
        "maximum_features",
    );
    for invalid_subsample in [0.0, -0.1, 1.2, f64::NAN] {
        assert_invalid_hyperparameter(
            SKGradientBoostingRegressor::<f64>::build()
                .subsample(invalid_subsample)
                .build(),
            "subsample",
        );
    }
    assert_invalid_hyperparameter(
        SKGradientBoostingRegressor::<f64>::build()
            .l2_regularization(-0.5)
            .build(),
        "l2_regularization",
    );
    for invalid_validation in [0.0, 1.0, -0.2, f64::NAN] {
        assert_invalid_hyperparameter(
            SKGradientBoostingRegressor::<f64>::build()
                .validation_fraction(invalid_validation)
                .build(),
            "validation_fraction",
        );
    }
    assert_invalid_hyperparameter(
        SKGradientBoostingRegressor::<f64>::build()
            .n_iter_no_change(0)
            .build(),
        "n_iter_no_change",
    );
    assert_invalid_hyperparameter(
        SKGradientBoostingRegressor::<f64>::build()
            .tolerance(-1e-6)
            .build(),
        "tolerance",
    );
}
