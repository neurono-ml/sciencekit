//! Companion tests for the single-sample update (spec
//! `stochastic-gradient-solver`, decision 3).

use ndarray::{Array1, array};

use super::config::SKSGDConfig;
use super::learning_rate::SKSGDLearningRate;
use super::learning_rate::schedule::SKLearningRateSchedule;
use super::loss::SKSGDRegressorLoss;
use super::loss::regressor::sk_regressor_loss_adapter;
use super::penalty::SKSGDPenalty;
use super::state::SKLearningState;
use super::update::sk_apply_single_sample_update;

fn build_state(
    weights: Vec<f64>,
    intercept: f64,
    config: &SKSGDConfig<f64>,
) -> SKLearningState<f64> {
    let schedule = SKLearningRateSchedule::new(
        config.learning_rate,
        config.initial_learning_rate,
        config.inverse_scaling_power,
        config.alpha,
        1.0,
    );
    SKLearningState::new(
        Array1::from_vec(weights),
        intercept,
        schedule,
        config.average,
    )
}

fn unpenalized_config() -> SKSGDConfig<f64> {
    let mut config = SKSGDConfig::<f64>::default();
    config.penalty = SKSGDPenalty::None;
    config.alpha = 0.0;
    config.learning_rate = SKSGDLearningRate::Constant;
    config.initial_learning_rate = 0.1;
    config
}

#[test]
fn one_update_equals_the_closed_form_step() {
    let config = unpenalized_config();
    let mut state = build_state(vec![1.0, 2.0], 0.0, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    // prediction 3, target 4 → slope −1; step 0.1.
    let weighted_loss = sk_apply_single_sample_update(
        &mut state,
        array![1.0, 1.0].view(),
        4.0,
        1.0,
        &loss,
        &config,
    );
    assert!((state.weights[0] - 1.1).abs() < 1e-12);
    assert!((state.weights[1] - 2.1).abs() < 1e-12);
    assert!((state.intercept - 0.1).abs() < 1e-12);
    assert!((weighted_loss - 0.5).abs() < 1e-12);
    assert_eq!(state.step, 1);
}

#[test]
fn the_sample_weight_scales_the_gradient() {
    let config = unpenalized_config();
    let mut state = build_state(vec![1.0, 2.0], 0.0, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    sk_apply_single_sample_update(
        &mut state,
        array![1.0, 1.0].view(),
        4.0,
        2.0,
        &loss,
        &config,
    );
    assert!((state.weights[0] - 1.2).abs() < 1e-12);
    assert!((state.intercept - 0.2).abs() < 1e-12);
}

#[test]
fn an_l1_penalty_produces_exact_zeros() {
    let mut config = unpenalized_config();
    config.penalty = SKSGDPenalty::L1;
    config.alpha = 1.0;
    config.initial_learning_rate = 1.0;
    let mut state = build_state(vec![0.05, -0.4], 0.0, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    sk_apply_single_sample_update(
        &mut state,
        array![0.0, 0.0].view(),
        1.0,
        1.0,
        &loss,
        &config,
    );
    assert_eq!(state.weights[0], 0.0);
    assert_eq!(state.weights[1], 0.0);
}

#[test]
fn the_intercept_is_never_penalized() {
    let mut config = unpenalized_config();
    config.penalty = SKSGDPenalty::L2;
    config.alpha = 0.5;
    config.initial_learning_rate = 1.0;
    let mut state = build_state(vec![2.0], 0.0, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    // The feature is zero, so only the penalty touches the weight: 2 → 1.
    sk_apply_single_sample_update(&mut state, array![0.0].view(), 1.0, 1.0, &loss, &config);
    assert!((state.weights[0] - 1.0).abs() < 1e-12);
    assert!((state.intercept - 1.0).abs() < 1e-12);
}

#[test]
fn a_zero_slope_is_a_no_op() {
    let config = unpenalized_config();
    let mut state = build_state(vec![1.0, 2.0], 0.5, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    // prediction 4.0, target 4.0 → slope zero.
    let weighted_loss = sk_apply_single_sample_update(
        &mut state,
        array![1.5, 1.0].view(),
        4.0,
        1.0,
        &loss,
        &config,
    );
    assert_eq!(state.weights, array![1.0, 2.0]);
    assert!((state.intercept - 0.5).abs() < 1e-12);
    assert_eq!(weighted_loss, 0.0);
}

#[test]
fn a_disabled_intercept_stays_zero() {
    let mut config = unpenalized_config();
    config.include_intercept = false;
    let mut state = build_state(vec![1.0], 0.0, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    sk_apply_single_sample_update(&mut state, array![1.0].view(), 4.0, 1.0, &loss, &config);
    assert_eq!(state.intercept, 0.0);
}

#[test]
fn averaging_reports_the_mean_of_the_visited_iterates() {
    let mut config = unpenalized_config();
    config.average = true;
    let mut state = build_state(vec![0.0], 0.0, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    sk_apply_single_sample_update(&mut state, array![1.0].view(), 1.0, 1.0, &loss, &config);
    let first = state.weights[0];
    sk_apply_single_sample_update(&mut state, array![1.0].view(), 3.0, 1.0, &loss, &config);
    let second = state.weights[0];
    assert_eq!(state.average_count(), 2);
    assert!((state.reported_weights()[0] - (first + second) / 2.0).abs() < 1e-12);
}

#[test]
fn the_update_counter_advances_the_schedule() {
    let mut config = unpenalized_config();
    config.learning_rate = SKSGDLearningRate::InverseScaling;
    let mut state = build_state(vec![0.0], 0.0, &config);
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    let first_rate = state.learning_rate();
    sk_apply_single_sample_update(&mut state, array![1.0].view(), 1.0, 1.0, &loss, &config);
    assert!(state.learning_rate() < first_rate);
}
