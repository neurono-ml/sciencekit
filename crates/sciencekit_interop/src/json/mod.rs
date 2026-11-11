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
mod parameter_deserialization;
mod reader;
mod reader_builder;
pub(crate) mod schema;
mod schema_deserialization;
mod state_decoding;
mod tensor_decoding;
mod writer;
mod writer_builder;

pub use format_mode::{SK_JSON_DEBUG_FORMAT_VERSION, SKJsonFormatMode};
pub use reader::{
    SKJsonDebugReader, sk_import_model_json, sk_read_model_state_json,
    sk_read_model_state_json_file,
};
pub use reader_builder::SKJsonDebugReaderBuilder;
pub use writer::{
    SKJsonDebugWriter, sk_export_model_json, sk_write_model_state_json,
    sk_write_model_state_json_compact, sk_write_model_state_json_file,
    sk_write_model_state_json_pretty,
};
pub use writer_builder::SKJsonDebugWriterBuilder;

#[cfg(test)]
mod json_round_trip_tests;
#[cfg(test)]
mod json_schema_tests;
#[cfg(test)]
mod json_writer_tests;
