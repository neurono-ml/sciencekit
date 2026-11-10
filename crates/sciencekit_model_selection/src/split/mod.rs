//! Train/test partitioning strategies (spec `model-selection-splits`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod splitter;

pub use splitter::SKSplitter;

#[cfg(test)]
mod splitter_tests;
