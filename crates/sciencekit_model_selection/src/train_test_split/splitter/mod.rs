//! The train/test splitter.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod core_implementation;
mod size_resolution;
mod split_implementation;

pub use core_implementation::SKTrainTestSplitter;
