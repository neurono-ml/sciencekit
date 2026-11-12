//! Shared fixtures for the streaming pipeline tests.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! the test transformers and sources; every implementation lives in its own
//! file.

mod test_sources;
mod test_transformers;

pub(crate) use test_sources::{
    SKTestBlocksSource, sk_reopenable_blocks, sk_single_block, sk_tiny_blocks,
};
pub(crate) use test_transformers::{SKTestColumnScaling, SKTestMeanCentering};
