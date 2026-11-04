//! Companion tests for the three epoch drivers (spec
//! `stochastic-gradient-solver`, decision 9).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKError, SKSupervisedBatch};
use sprs::{CsMat, TriMat};

use super::{sk_run_batch_epoch, sk_run_dense_epoch, sk_run_sparse_epoch};
use crate::sgd::config::SKSGDConfig;
use crate::sgd::learning_rate::SKSGDLearningRate;
use crate::sgd::learning_rate::schedule::SKLearningRateSchedule;
use crate::sgd::loss::SKSGDRegressorLoss;
use crate::sgd::loss::regressor::sk_regressor_loss_adapter;
use crate::sgd::penalty::SKSGDPenalty;
use crate::sgd::state::SKLearningState;

fn config() -> SKSGDConfig<f64> {
    let mut config = SKSGDConfig::<f64>::default();
    config.penalty = SKSGDPenalty::None;
    config.alpha = 0.0;
    config.learning_rate = SKSGDLearningRate::Constant;
    config.initial_learning_rate = 0.1;
    config
}

fn state(config: &SKSGDConfig<f64>, columns: usize) -> SKLearningState<f64> {
    let schedule = SKLearningRateSchedule::new(
        config.learning_rate,
        config.initial_learning_rate,
        config.inverse_scaling_power,
        config.alpha,
        1.0,
    );
    SKLearningState::new(Array1::zeros(columns), 0.0, schedule, config.average)
}

fn to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense.indexed_iter().for_each(|((row, column), &value)| {
        if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

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
fn the_sparse_driver_never_densifies() {
    // A wide matrix with two stored values: only the touched columns may move.
    let mut triplets = TriMat::new((1, 1000));
    triplets.add_triplet(0, 3, 1.0);
    triplets.add_triplet(0, 999, 2.0);
    let sparse = triplets.to_csr();
    let config = config();
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    let mut state = state(&config, 1000);
    let objective = sk_run_sparse_epoch(
        &mut state,
        sparse.view(),
        array![1.0].view(),
        &[1.0],
        &loss,
        &config,
        &[0],
    )
    .unwrap();
    assert_eq!(state.weights[3], 0.1);
    assert_eq!(state.weights[999], 0.2);
    assert_eq!(state.weights[0], 0.0);
    assert_eq!(state.weights[500], 0.0);
    assert_eq!(objective.count(), 1);
}

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
fn shape_mismatches_are_rejected() {
    let design = array![[1.0], [2.0]];
    let config = config();
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    let mut state = state(&config, 1);

    let targets = array![1.0];
    let error = sk_run_dense_epoch(
        &mut state,
        design.view(),
        targets.view(),
        &[1.0],
        &loss,
        &config,
        &[0],
    )
    .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));

    let targets = array![1.0, 2.0];
    let error = sk_run_dense_epoch(
        &mut state,
        design.view(),
        targets.view(),
        &[1.0],
        &loss,
        &config,
        &[0, 1],
    )
    .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));

    let error = sk_run_dense_epoch(
        &mut state,
        design.view(),
        targets.view(),
        &[1.0, 1.0],
        &loss,
        &config,
        &[0, 1, 2],
    )
    .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
