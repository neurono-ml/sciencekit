//! Mergeable streaming accumulators for batched and parallel evaluation.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classification;

pub use classification::SKClassificationAccumulator;

#[cfg(test)]
mod classification_tests;
