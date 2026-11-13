//! Train/test splitting (spec `model-selection-splits`, requirements
//! `Train/test split` and `Stratified split`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod configuration;
mod partition_size;
mod sk_train_test_split;
mod split;
mod splitter;
mod stratification;

pub use builder::SKTrainTestSplitBuilder;
pub use partition_size::SKPartitionSize;
pub use sk_train_test_split::sk_train_test_split;
pub use split::SKTrainTestSplit;
pub use splitter::SKTrainTestSplitter;
pub(crate) use stratification::sk_allocate_class_counts;

#[cfg(test)]
mod stratified_tests;
#[cfg(test)]
mod train_test_split_tests;
