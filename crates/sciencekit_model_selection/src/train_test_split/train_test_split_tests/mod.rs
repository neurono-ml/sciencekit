//! Train/test split fraction, count, shuffle, gathering and validation tests
//! (spec `model-selection-splits`, requirement `Train/test split`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules; the
//! scenarios live in their own files.

mod fixtures;
mod gathering_tests;
mod rejection_tests;
mod size_tests;
