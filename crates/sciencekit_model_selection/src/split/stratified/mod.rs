//! Stratified K-fold: proportions preserved (spec `model-selection-splits`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod configuration;
mod splitter;

pub use builder::SKStratifiedKFoldBuilder;
pub use configuration::SKStratifiedKFoldConfiguration;
pub use splitter::SKStratifiedKFold;

#[cfg(test)]
mod stratified_tests;
