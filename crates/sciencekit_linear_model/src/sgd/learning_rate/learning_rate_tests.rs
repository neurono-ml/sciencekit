//! Companion tests for the learning-rate schedules (spec
//! `stochastic-gradient-solver`, decision 4).

#![allow(clippy::field_reassign_with_default)]

use sciencekit_common::SKError;

use super::schedule::SKLearningRateSchedule;
use super::{SKSGDLearningRate, optimal::sk_optimal_offset};
use crate::sgd::config::{SKSGDConfig, sk_validate_sgd_configuration};

fn schedule(policy: SKSGDLearningRate, initial: f64) -> SKLearningRateSchedule<f64> {
    SKLearningRateSchedule::new(policy, initial, 0.5, 0.5, 2.0)
}

#[test]
fn constant_keeps_the_configured_rate() {
    let schedule = schedule(SKSGDLearningRate::Constant, 0.7);
    assert!((schedule.rate(0) - 0.7).abs() < 1e-12);
    assert!((schedule.rate(500) - 0.7).abs() < 1e-12);
}

#[test]
fn inverse_scaling_decays_monotonically() {
    let schedule = schedule(SKSGDLearningRate::InverseScaling, 1.0);
    assert!((schedule.rate(0) - 1.0).abs() < 1e-12);
    let mut previous = schedule.rate(0);
    for step in 1..50 {
        let current = schedule.rate(step);
        assert!(current < previous, "the rate must decay at step {step}");
        previous = current;
    }
    // η_t = η₀ / (1 + t)^p with η₀ = 1, p = 0.5 at t = 3.
    assert!((schedule.rate(3) - 4.0_f64.sqrt().recip()).abs() < 1e-12);
}

#[test]
fn optimal_follows_the_inverse_strength_form() {
    let schedule = schedule(SKSGDLearningRate::Optimal, 999.0);
    // The initial rate is not consulted: 1 / (0.5 · (t + 2)).
    assert!((schedule.rate(0) - 1.0).abs() < 1e-12);
    assert!((schedule.rate(2) - 0.5).abs() < 1e-12);
    assert!((schedule.rate(8) - 0.2).abs() < 1e-12);
}

#[test]
fn the_optimal_offset_is_at_least_one() {
    assert!((sk_optimal_offset(0.5_f64, 1.0) - 2.0).abs() < 1e-12);
    assert!((sk_optimal_offset(2.0_f64, 1.0) - 1.0).abs() < 1e-12);
    assert!((sk_optimal_offset(2.0_f64, 0.0) - 1.0).abs() < 1e-12);
}

#[test]
fn the_optimal_schedule_rejects_zero_alpha_at_build() {
    let mut config = SKSGDConfig::<f64>::default();
    config.learning_rate = SKSGDLearningRate::Optimal;
    config.alpha = 0.0;
    let error = sk_validate_sgd_configuration(&config).unwrap_err();
    assert!(
        matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "learning_rate")
    );
}

#[test]
fn adaptive_halves_on_a_plateau_and_never_grows() {
    let mut schedule = schedule(SKSGDLearningRate::Adaptive, 0.8);
    assert!((schedule.rate(10) - 0.8).abs() < 1e-12);
    schedule.halve();
    assert!((schedule.rate(11) - 0.4).abs() < 1e-12);
    schedule.halve();
    assert!((schedule.rate(12) - 0.2).abs() < 1e-12);
    schedule.halve();
    assert!(schedule.rate(13) <= schedule.rate(12));
}

#[test]
fn halving_is_a_no_op_for_the_other_schedules() {
    for policy in [
        SKSGDLearningRate::Constant,
        SKSGDLearningRate::Optimal,
        SKSGDLearningRate::InverseScaling,
    ] {
        let mut schedule = schedule(policy, 1.0);
        let before = schedule.rate(4);
        schedule.halve();
        assert!((schedule.rate(4) - before).abs() < 1e-12);
    }
}

#[test]
fn the_counter_reflects_the_total_number_of_updates() {
    // A partial fit that follows an earlier one continues the same counter:
    // the rate after ten updates is the formula evaluated at ten, not at zero.
    let schedule = schedule(SKSGDLearningRate::InverseScaling, 1.0);
    let after_ten_updates = schedule.rate(10);
    assert!((after_ten_updates - (11.0_f64).sqrt().recip()).abs() < 1e-12);
    assert!(after_ten_updates < schedule.rate(0));
}
