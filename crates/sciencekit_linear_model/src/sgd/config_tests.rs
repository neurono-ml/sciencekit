//! Companion tests for the stochastic-gradient configuration validation (spec
//! `stochastic-gradient-solver`).

use sciencekit_common::SKError;

use super::config::{SKSGDConfig, sk_validate_sgd_configuration};
use super::learning_rate::SKSGDLearningRate;

fn assert_rejects(config: SKSGDConfig<f64>, name: &'static str) {
    let error = sk_validate_sgd_configuration(&config).unwrap_err();
    assert!(
        matches!(error, SKError::InvalidHyperparameter { name: actual, .. } if actual == name),
        "expected an error naming `{name}`, got {error:?}"
    );
}

#[test]
fn a_well_formed_configuration_is_accepted() {
    assert!(sk_validate_sgd_configuration(&SKSGDConfig::<f64>::default()).is_ok());
}

#[test]
fn zero_alpha_is_accepted_but_a_negative_one_is_rejected() {
    let mut config = SKSGDConfig::<f64>::default();
    config.alpha = 0.0;
    assert!(sk_validate_sgd_configuration(&config).is_ok());

    config.alpha = -0.1;
    assert_rejects(config, "alpha");
}

#[test]
fn an_out_of_range_l1_ratio_is_rejected() {
    let mut config = SKSGDConfig::<f64>::default();
    config.l1_ratio = 1.5;
    assert_rejects(config, "l1_ratio");

    let mut config = SKSGDConfig::<f64>::default();
    config.l1_ratio = -0.1;
    assert_rejects(config, "l1_ratio");
}

#[test]
fn a_non_positive_tolerance_is_rejected() {
    let mut config = SKSGDConfig::<f64>::default();
    config.tolerance = 0.0;
    assert_rejects(config, "tolerance");
}

#[test]
fn a_zero_iteration_budget_is_rejected() {
    let mut config = SKSGDConfig::<f64>::default();
    config.maximum_number_of_iterations = 0;
    assert_rejects(config, "maximum_number_of_iterations");
}

#[test]
fn a_non_positive_initial_rate_is_rejected_when_used() {
    for schedule in [
        SKSGDLearningRate::Constant,
        SKSGDLearningRate::InverseScaling,
        SKSGDLearningRate::Adaptive,
    ] {
        let mut config = SKSGDConfig::<f64>::default();
        config.learning_rate = schedule;
        config.initial_learning_rate = 0.0;
        assert_rejects(config, "initial_learning_rate");
    }
}

#[test]
fn the_optimal_schedule_does_not_consult_the_initial_rate() {
    let mut config = SKSGDConfig::<f64>::default();
    config.learning_rate = SKSGDLearningRate::Optimal;
    config.initial_learning_rate = 0.0;
    config.alpha = 0.5;
    assert!(sk_validate_sgd_configuration(&config).is_ok());
}

#[test]
fn a_non_positive_inverse_scaling_power_is_rejected() {
    let mut config = SKSGDConfig::<f64>::default();
    config.learning_rate = SKSGDLearningRate::InverseScaling;
    config.inverse_scaling_power = 0.0;
    assert_rejects(config, "inverse_scaling_power");
}

#[test]
fn a_validation_fraction_outside_the_open_interval_is_rejected() {
    for fraction in [0.0, 1.0, 1.5] {
        let mut config = SKSGDConfig::<f64>::default();
        config.validation_fraction = fraction;
        assert_rejects(config, "validation_fraction");
    }
}

#[test]
fn zero_epochs_without_improvement_is_rejected() {
    let mut config = SKSGDConfig::<f64>::default();
    config.number_of_epochs_without_improvement = 0;
    assert_rejects(config, "number_of_epochs_without_improvement");
}

#[test]
fn optimal_requires_a_positive_alpha() {
    let mut config = SKSGDConfig::<f64>::default();
    config.learning_rate = SKSGDLearningRate::Optimal;
    config.alpha = 0.0;
    assert_rejects(config, "learning_rate");
}

#[test]
fn early_stopping_requires_more_than_one_epoch() {
    let mut config = SKSGDConfig::<f64>::default();
    config.early_stopping = true;
    config.maximum_number_of_iterations = 1;
    assert_rejects(config, "early_stopping");
}

#[test]
fn an_explicit_learning_rate_is_not_collapsed_by_default() {
    let config = SKSGDConfig::<f64>::default();
    assert!(config.learning_rate.uses_initial_rate());
}
