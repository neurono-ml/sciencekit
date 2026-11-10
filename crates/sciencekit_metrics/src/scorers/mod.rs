//! Scorer adapters implementing the shared scoring contracts.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classification;
mod ranking;
mod regression;
mod targets;

pub use classification::{SKAccuracyScorer, SKF1Scorer, SKPrecisionScorer, SKRecallScorer};
pub use ranking::{
    SKAucScorer, SKAveragePrecisionScorer, SKBalancedAccuracyScorer, SKLogLossScorer,
    SKMatthewsCorrelationScorer,
};
pub use regression::{
    SKMeanAbsoluteErrorScorer, SKMeanSquaredErrorScorer, SKR2Scorer, SKRootMeanSquaredErrorScorer,
};

#[cfg(test)]
mod classification_tests;
#[cfg(test)]
mod ranking_scorers_tests;
#[cfg(test)]
mod regression_tests;
