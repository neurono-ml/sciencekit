//! Cross-cutting acceptance scenarios (spec `model-selection-splits`, PRD §8.7).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules; the
//! scenarios live in their own files.

mod concurrency_tests;
mod determinism_tests;
mod fixtures;
mod grid_search;
mod halving;
mod large_data_tests;
mod metric_tests;
mod representation_tests;
mod tiny_data_tests;
