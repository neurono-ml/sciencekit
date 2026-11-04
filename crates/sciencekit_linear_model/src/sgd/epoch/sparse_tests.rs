//! Companion tests for the sparse epoch driver's zero-copy behaviour (spec
//! `stochastic-gradient-solver`, decision 9).

use ndarray::array;
use sprs::TriMat;

use super::sk_run_sparse_epoch;
use super::test_helpers::{config, state};
use crate::sgd::loss::SKSGDRegressorLoss;
use crate::sgd::loss::regressor::sk_regressor_loss_adapter;

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
