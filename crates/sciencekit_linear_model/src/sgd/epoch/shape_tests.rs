//! Companion tests for the epoch drivers' shape checks (spec
//! `stochastic-gradient-solver`, decision 9).

use ndarray::array;
use sciencekit_common::SKError;

use super::sk_run_dense_epoch;
use super::test_helpers::{config, state};
use crate::sgd::loss::SKSGDRegressorLoss;
use crate::sgd::loss::regressor::sk_regressor_loss_adapter;

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

    // An order may be a strict subset of the rows (early stopping reserves a
    // validation prefix), but every visited index must still be in range.
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

    let subset_objective = sk_run_dense_epoch(
        &mut state,
        design.view(),
        targets.view(),
        &[1.0, 1.0],
        &loss,
        &config,
        &[1],
    )
    .unwrap();
    assert_eq!(subset_objective.count(), 1);
}
