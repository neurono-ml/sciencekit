//! Differentiable loss functions for stage-wise gradient boosting.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod absolute;
mod contract;
mod huber;
mod squared;

pub use absolute::SKAbsoluteErrorLoss;
pub use contract::SKGradientLoss;
pub use huber::{SKHuberLoss, SKHuberLossBuilder};
pub use squared::SKSquaredErrorLoss;

#[cfg(test)]
mod boosting_loss_tests;
