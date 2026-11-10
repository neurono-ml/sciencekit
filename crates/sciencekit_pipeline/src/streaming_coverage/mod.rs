//! Streaming coverage audit (spec `sparse-streaming-coverage`).
//!
//! This is a **pure dispatcher** module: `mod.rs` declares and re-exports the
//! coverage matrix, the audit tests and the dense-only guard fixture.

mod guard;
mod matrix;
mod source_scan;

#[cfg(test)]
mod matrix_tests;
