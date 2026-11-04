//! Companion tests for the dense and sparse epoch drivers (spec
//! `stochastic-gradient-solver`, decision 9).

#![allow(clippy::field_reassign_with_default)]

use ndarray::array;

use super::sk_run_dense_epoch;
use super::sk_run_sparse_epoch;
use super::test_helpers::{config, state, to_csr};
use crate::sgd::loss::SKSGDRegressorLoss;
use crate::sgd::loss::regressor::sk_regressor_loss_adapter;

#[test]
fn dense_and_sparse_agree_under_the_same_order() {
    let design = array![
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 3.0, 1.0],
    ];
    let targets = array![1.0, -1.0, 3.0, 2.0];
    let weights = [1.0, 0.5, 2.0, 1.0];
    let order = [3, 0, 2, 1];
    let config = config();
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);

    let mut dense_state = state(&config, 3);
    let dense_objective = sk_run_dense_epoch(
        &mut dense_state,
        design.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &order,
    )
    .unwrap();

    let sparse = to_csr(&design);
    let mut sparse_state = state(&config, 3);
    let sparse_objective = sk_run_sparse_epoch(
        &mut sparse_state,
        sparse.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &order,
    )
    .unwrap();

    for column in 0..3 {
        assert!((dense_state.weights[column] - sparse_state.weights[column]).abs() < 1e-12);
    }
    assert!((dense_state.intercept - sparse_state.intercept).abs() < 1e-12);
    assert!((dense_objective.mean() - sparse_objective.mean()).abs() < 1e-12);
    assert_eq!(dense_state.step, 4);
}

#[test]
fn the_order_controls_the_visit_sequence() {
    let features = array![[1.0], [1.0]];
    let targets = array![1.0, 3.0];
    let config = config();
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);

    let mut forward = state(&config, 1);
    sk_run_dense_epoch(
        &mut forward,
        features.view(),
        targets.view(),
        &[1.0, 1.0],
        &loss,
        &config,
        &[0, 1],
    )
    .unwrap();
    let mut reverse = state(&config, 1);
    sk_run_dense_epoch(
        &mut reverse,
        features.view(),
        targets.view(),
        &[1.0, 1.0],
        &loss,
        &config,
        &[1, 0],
    )
    .unwrap();
    assert!((forward.weights[0] - reverse.weights[0]).abs() > 1e-6);
}

#[test]
fn dense_and_sparse_agree_on_f32() {
    let design = array![[1.0_f32, 0.0], [0.0, 2.0], [1.0, 1.0]];
    let targets = array![1.0_f32, 2.0, 3.0];
    let weights = [1.0_f32, 1.0, 1.0];
    let order = [2, 0, 1];
    let mut config = crate::sgd::config::SKSGDConfig::<f32>::default();
    config.penalty = crate::sgd::penalty::SKSGDPenalty::None;
    config.alpha = 0.0;
    config.learning_rate = crate::sgd::learning_rate::SKSGDLearningRate::Constant;
    config.initial_learning_rate = 0.1;
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0_f32);

    let make_state = || {
        let schedule = crate::sgd::learning_rate::schedule::SKLearningRateSchedule::new(
            config.learning_rate,
            config.initial_learning_rate,
            config.inverse_scaling_power,
            config.alpha,
            1.0,
        );
        crate::sgd::state::SKLearningState::new(
            ndarray::Array1::zeros(2),
            0.0,
            schedule,
            config.average,
        )
    };
    let mut dense_state = make_state();
    sk_run_dense_epoch(
        &mut dense_state,
        design.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &order,
    )
    .unwrap();
    let sparse = to_csr(&design);
    let mut sparse_state = make_state();
    sk_run_sparse_epoch(
        &mut sparse_state,
        sparse.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &order,
    )
    .unwrap();
    for column in 0..2 {
        assert!((dense_state.weights[column] - sparse_state.weights[column]).abs() < 1e-6);
    }
    assert!((dense_state.intercept - sparse_state.intercept).abs() < 1e-6);
}
