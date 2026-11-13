//! Zero-copy feature views (spec `dataset-interop-core`).
//!
//! A conversion borrows only when the input already matches the row-major
//! layout; everything else materializes a fresh block. This module holds the
//! borrowed shapes.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod zero_copy_views;

pub use zero_copy_views::{SKNumericColumnView, SKRowMajorFeatureView};

#[cfg(test)]
mod views_tests;
