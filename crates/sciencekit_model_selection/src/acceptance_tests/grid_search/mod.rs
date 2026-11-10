//! Grid-search acceptance scenarios (spec `grid-search`, PRD §8.7).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules; the
//! scenarios live in their own files.

mod large_data_tests;
mod metric_tests;
mod scoring_policy_tests;
mod splitter_tests;
mod tiny_data_tests;
