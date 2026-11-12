//! Successive-halving grid search acceptance scenarios
//! (spec `halving-grid-search`, PRD §8.7).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules; the
//! scenarios live in their own files.

mod large_data_tests;
mod metric_and_determinism_tests;
mod splitter_and_aggressive_tests;
mod tiny_data_tests;
mod validation_tests;
