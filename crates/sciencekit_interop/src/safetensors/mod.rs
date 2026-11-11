//! Extended Safetensors internal model format over [`sciencekit_common::SKModelState`]
//! (PRD §8.2, spec `safetensors-format`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod header;
mod parameter_codec;
mod reader;
mod reader_builder;
mod single_file;
mod tensor_codec;
mod writer;
mod writer_builder;

pub use header::{SK_SAFETENSORS_FORMAT_VERSION, SK_SAFETENSORS_PADDING_TENSOR_NAME};
pub use reader::{
    SKSafetensorsReader, sk_import_model_safetensors, sk_import_model_safetensors_file,
    sk_read_model_state_safetensors, sk_read_model_state_safetensors_file,
};
pub use reader_builder::SKSafetensorsReaderBuilder;
pub use writer::{
    SKSafetensorsWriter, sk_export_model_safetensors, sk_export_model_safetensors_file,
    sk_write_model_state_safetensors, sk_write_model_state_safetensors_file,
};
pub use writer_builder::SKSafetensorsWriterBuilder;

#[cfg(test)]
mod safetensors_round_trip_tests;
#[cfg(test)]
mod safetensors_wiring_tests;
