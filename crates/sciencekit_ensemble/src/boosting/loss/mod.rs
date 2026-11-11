//! Differentiable loss functions for stage-wise gradient boosting.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod absolute;
mod contract;
mod huber;
mod log_loss;
mod squared;

pub use absolute::SKAbsoluteErrorLoss;
pub use contract::SKGradientLoss;
pub use huber::{SKHuberLoss, SKHuberLossBuilder};
pub use log_loss::{SKBinaryLogLoss, SKMulticlassLogLoss, SKMulticlassLogLossBuilder};
pub use squared::SKSquaredErrorLoss;

#[cfg(test)]
mod boosting_log_loss_tests;
#[cfg(test)]
mod boosting_loss_tests;
