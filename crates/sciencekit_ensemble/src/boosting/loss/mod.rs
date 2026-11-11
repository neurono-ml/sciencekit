//! Differentiable loss functions for stage-wise gradient boosting.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod contract;
mod squared;

pub use contract::SKGradientLoss;
pub use squared::SKSquaredErrorLoss;

#[cfg(test)]
mod boosting_loss_tests;
