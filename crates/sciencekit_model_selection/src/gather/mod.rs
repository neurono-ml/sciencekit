//! Row gathering for feature blocks and targets (spec `model-selection-splits`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod feature_gathering;
mod rows;
mod target_gathering;
mod target_partition;

pub use rows::sk_gather_rows;
pub use target_partition::SKTargetPartition;

pub(crate) use feature_gathering::sk_gather_feature_rows;
pub(crate) use target_gathering::sk_gather_target_rows;

#[cfg(test)]
mod gather_tests;
