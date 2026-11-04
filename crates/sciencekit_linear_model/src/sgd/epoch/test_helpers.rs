//! Shared helpers for the epoch-driver companion tests.

#![allow(clippy::field_reassign_with_default)]

use ndarray::{Array1, Array2};
use sciencekit_common::SKFloat;
use sprs::{CsMat, TriMat};

use crate::sgd::config::SKSGDConfig;
use crate::sgd::learning_rate::SKSGDLearningRate;
use crate::sgd::learning_rate::schedule::SKLearningRateSchedule;
use crate::sgd::penalty::SKSGDPenalty;
use crate::sgd::state::SKLearningState;

/// An unpenalized constant-rate configuration with a live intercept.
pub(crate) fn config() -> SKSGDConfig<f64> {
    let mut config = SKSGDConfig::<f64>::default();
    config.penalty = SKSGDPenalty::None;
    config.alpha = 0.0;
    config.learning_rate = SKSGDLearningRate::Constant;
    config.initial_learning_rate = 0.1;
    config
}

/// A zero-initialized learning state with `columns` features.
pub(crate) fn state(config: &SKSGDConfig<f64>, columns: usize) -> SKLearningState<f64> {
    let schedule = SKLearningRateSchedule::new(
        config.learning_rate,
        config.initial_learning_rate,
        config.inverse_scaling_power,
        config.alpha,
        1.0,
    );
    SKLearningState::new(Array1::zeros(columns), 0.0, schedule, config.average)
}

/// Convert a dense design to an equivalent CSR matrix, dropping exact zeros.
pub(crate) fn to_csr<F: SKFloat>(dense: &Array2<F>) -> CsMat<F> {
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense.indexed_iter().for_each(|((row, column), &value)| {
        if value != F::zero() {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}
