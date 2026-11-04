//! The loss catalogue of the stochastic-gradient substrate
//! (spec `stochastic-gradient-solver`, decision 2).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier;
pub(crate) mod definition;
mod regressor;

pub use classifier::SKSGDClassifierLoss;
pub use definition::SKSGDLoss;
pub use regressor::SKSGDRegressorLoss;

#[cfg(test)]
mod loss_tests;
