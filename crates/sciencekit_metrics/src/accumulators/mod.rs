//! Mergeable streaming accumulators for batched and parallel evaluation.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.


#[cfg(test)]
mod classification_tests;
#[cfg(test)]
mod regression_tests;

pub mod concurrency_tests;

mod classification;
mod regression;
pub use classification::SKClassificationAccumulator;
pub use regression::SKRegressionAccumulator;
