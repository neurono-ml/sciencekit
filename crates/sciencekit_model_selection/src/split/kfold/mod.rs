//! Plain K-fold: disjoint, exhaustive and seeded (spec `model-selection-splits`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod configuration;
mod splitter;

pub use builder::SKKFoldBuilder;
pub use configuration::SKKFoldConfiguration;
pub use splitter::SKKFold;

#[cfg(test)]
mod kfold_tests;
