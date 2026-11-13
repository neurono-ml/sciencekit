//! The owned streaming feature block (spec `sparse-streaming-batches`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod block;
mod conversions;

pub use block::SKFeatureBlock;
pub use conversions::SKIntoFeatureBlock;

#[cfg(test)]
mod conversions_tests;
#[cfg(test)]
mod feature_block_tests;
