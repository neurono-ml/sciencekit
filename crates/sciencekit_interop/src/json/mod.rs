//! Human-readable JSON debug representation of [`sciencekit_common::SKModelState`]
//! (PRD §8.6, spec `json-debug-format`).
//!
//! # Memory behavior for large tensors
//!
//! Unlike the binary Safetensors format, JSON expands every numeric element into
//! UTF-8 decimal text and materializes flat value arrays in memory during
//! serialization and parsing. The JSON debug format is intended for human
//! inspection, clean line diffs and test fixtures rather than multi-gigabyte
//! production checkpoints.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod format_mode;
pub(crate) mod schema;

pub use format_mode::{SK_JSON_DEBUG_FORMAT_VERSION, SKJsonFormatMode};

#[cfg(test)]
mod json_schema_tests;
