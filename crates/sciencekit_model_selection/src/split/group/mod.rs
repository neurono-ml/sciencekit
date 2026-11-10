//! Group K-fold: group integrity preserved (spec `model-selection-splits`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod configuration;
mod splitter;

pub use builder::SKGroupKFoldBuilder;
pub use configuration::SKGroupKFoldConfiguration;
pub use splitter::SKGroupKFold;

#[cfg(test)]
mod group_tests;
