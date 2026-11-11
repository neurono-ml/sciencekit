//! Shared gradient-boosting substrate: loss contract, initial predictions,
//! stage fitting, subsampling and early stopping (spec `gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

pub mod loss;

pub use loss::{
    SKAbsoluteErrorLoss, SKBinaryLogLoss, SKGradientLoss, SKHuberLoss, SKHuberLossBuilder,
    SKMulticlassLogLoss, SKMulticlassLogLossBuilder, SKSquaredErrorLoss,
};
