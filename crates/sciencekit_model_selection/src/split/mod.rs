//! Train/test partitioning strategies (spec `model-selection-splits`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod group;
pub mod kfold;
mod splitter;
pub mod stratified;
pub(crate) mod validation;

pub use splitter::SKSplitter;

#[cfg(test)]
mod splitter_tests;
