//! Shared gradient-boosting substrate: loss contract, initial predictions,
//! stage fitting, subsampling and early stopping (spec `gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod init;
pub mod loss;
pub(crate) mod scores;
pub(crate) mod stage;
pub(crate) mod subsample;

pub use init::{
    sk_initialize_multiclass_scores, sk_initialize_scalar_scores, sk_newton_leaf_value,
};
pub use loss::{
    SKAbsoluteErrorLoss, SKBinaryLogLoss, SKGradientLoss, SKHuberLoss, SKHuberLossBuilder,
    SKMulticlassLogLoss, SKMulticlassLogLossBuilder, SKSquaredErrorLoss,
};

#[cfg(test)]
mod boosting_stage_tests;
#[cfg(test)]
mod boosting_subsample_tests;
