//! The shared stochastic-gradient substrate of the linear models
//! (spec `stochastic-gradient-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file. The substrate is
//! consumed by the classifier and regressor estimators in the same crate, so
//! its internal pieces are crate-visible while the loss catalogue, the schedule
//! selection, the penalty selection and the convergence policy are public.

#![allow(dead_code)]

mod averaging;
pub(crate) mod config;
mod epoch;
mod learning_rate;
mod loss;
pub(crate) mod objective;
mod penalty;
mod row;
mod shuffle;
mod state;
mod update;

pub use learning_rate::SKSGDLearningRate;
pub use loss::{SKSGDClassifierLoss, SKSGDLoss, SKSGDRegressorLoss};
pub use penalty::SKSGDPenalty;

#[cfg(test)]
mod averaging_tests;
#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod objective_tests;
#[cfg(test)]
mod penalty_tests;
#[cfg(test)]
mod shuffle_tests;
#[cfg(test)]
mod update_tests;
