//! The learning-rate schedules of the stochastic-gradient substrate
//! (spec `stochastic-gradient-solver`, decision 4).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod optimal;
mod policy;
pub(crate) mod schedule;

pub use policy::SKSGDLearningRate;

#[cfg(test)]
mod learning_rate_tests;
