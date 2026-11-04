//! Companion tests for the streaming-batch epoch driver (spec
//! `stochastic-gradient-solver`, decision 9).

use ndarray::array;
use sciencekit_common::SKSupervisedBatch;

use super::sk_run_batch_epoch;
use super::sk_run_dense_epoch;
use super::test_helpers::{config, state};
use crate::sgd::loss::SKSGDRegressorLoss;
use crate::sgd::loss::regressor::sk_regressor_loss_adapter;

#[test]
fn the_batch_driver_visits_exactly_its_block() {
    let features = array![[1.0], [2.0], [3.0]];
    let targets = array![1.0, 2.0, 3.0];
    let batch = SKSupervisedBatch::new(features.clone(), targets.clone(), 0, true);
    let config = config();
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);

    let mut batch_state = state(&config, 1);
    let batch_objective = sk_run_batch_epoch(
        &mut batch_state,
        &batch,
        &[1.0, 1.0, 1.0],
        &loss,
        &config,
        &[0, 1, 2],
    )
    .unwrap();

    let mut dense_state = state(&config, 1);
    let dense_objective = sk_run_dense_epoch(
        &mut dense_state,
        features.view(),
        targets.view(),
        &[1.0, 1.0, 1.0],
        &loss,
        &config,
        &[0, 1, 2],
    )
    .unwrap();

    assert_eq!(batch_objective.count(), 3);
    assert!((batch_state.weights[0] - dense_state.weights[0]).abs() < 1e-12);
    assert!((batch_objective.mean() - dense_objective.mean()).abs() < 1e-12);
}
