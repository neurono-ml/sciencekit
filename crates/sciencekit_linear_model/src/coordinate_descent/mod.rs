//! The shared coordinate-descent substrate for the L1-regularized linear models
//! (spec `coordinate-descent-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub mod config;

pub use config::{SKIntoAlphas, SKSelection};

#[cfg(test)]
mod config_tests;
