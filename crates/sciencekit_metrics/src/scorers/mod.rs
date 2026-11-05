//! Scorer adapters implementing the shared scoring contracts.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classification;
mod targets;

pub use classification::{SKAccuracyScorer, SKF1Scorer, SKPrecisionScorer, SKRecallScorer};

#[cfg(test)]
mod classification_tests;
