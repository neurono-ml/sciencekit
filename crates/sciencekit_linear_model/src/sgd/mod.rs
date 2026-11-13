//! The shared stochastic-gradient substrate of the linear models
//! (spec `stochastic-gradient-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file. The substrate is
//! consumed by the classifier and regressor estimators in the same crate, so
//! its internal pieces are crate-visible while the loss catalogue, the schedule
//! selection, the penalty selection and the convergence policy are public.

mod averaging;
pub(crate) mod config;
mod convergence;
mod early_stopping;
mod epoch;
mod learning_rate;
mod loss;
pub(crate) mod objective;
mod penalty;
mod row;
mod state;
mod update;

pub use crate::class_weight::SKClassWeight;
pub use convergence::SKConvergencePolicy;
pub use learning_rate::SKSGDLearningRate;
pub use loss::{SKSGDClassifierLoss, SKSGDLoss, SKSGDRegressorLoss};
pub use penalty::SKSGDPenalty;

// Crate-visible plumbing the estimators drive the substrate with. The pieces
// stay out of the public surface: only the learning loop, the drivers and the
// bookkeeping the estimator owns are re-exported.
pub(crate) use convergence::{SKConvergenceTracker, sk_finish_epochs};
pub(crate) use early_stopping::{SKEarlyStopping, sk_split_validation_indices};
pub(crate) use epoch::{
    sk_run_batch_epoch, sk_run_block_epoch, sk_run_dense_epoch, sk_run_sparse_epoch,
};
pub(crate) use learning_rate::schedule::SKLearningRateSchedule;
pub(crate) use learning_rate::sk_optimal_offset;
pub(crate) use loss::regressor::{SKSGDRegressorLossAdapter, sk_regressor_loss_adapter};
pub(crate) use objective::sk_objective_improved;
pub(crate) use state::SKLearningState;

#[cfg(test)]
mod averaging_tests;
#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod convergence_tests;
#[cfg(test)]
mod early_stopping_tests;
#[cfg(test)]
mod objective_tests;
#[cfg(test)]
mod penalty_tests;
#[cfg(test)]
mod shuffle_tests;
#[cfg(test)]
mod update_penalty_tests;
#[cfg(test)]
mod update_tests;
