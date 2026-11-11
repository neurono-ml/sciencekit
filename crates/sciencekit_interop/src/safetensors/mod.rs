//! Extended Safetensors internal model format over [`sciencekit_common::SKModelState`]
//! (PRD §8.2, spec `safetensors-format`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod header;
mod padding_append;
mod padding_encode;
mod parameter_codec;
mod reader;
mod reader_builder;
mod shard_index;
mod shard_io;
mod shard_plan;
mod single_file;
mod tensor_codec;
mod wire_header_deserialization;
mod wire_validation;
mod writer;
mod writer_builder;
mod writer_functions;

pub use header::{SK_SAFETENSORS_FORMAT_VERSION, SK_SAFETENSORS_PADDING_TENSOR_NAME};
pub use padding_encode::SKAppendOutcome;
pub use reader::{
    SKSafetensorsReader, sk_import_model_safetensors, sk_import_model_safetensors_file,
    sk_read_model_state_safetensors, sk_read_model_state_safetensors_file,
};
pub use reader_builder::SKSafetensorsReaderBuilder;
pub use shard_index::{sk_read_safetensors_shard_index, sk_safetensors_index_path};
pub use shard_plan::{
    SK_DEFAULT_MAXIMUM_SHARD_BYTES, SKShardIndex, SKShardPlan, SKShardPlanBuilder,
};
pub use writer::SKSafetensorsWriter;
pub use writer_builder::SKSafetensorsWriterBuilder;
pub use writer_functions::{
    sk_append_safetensors_tensor, sk_export_model_safetensors, sk_export_model_safetensors_file,
    sk_write_model_state_safetensors, sk_write_model_state_safetensors_file,
};

#[cfg(test)]
mod safetensors_padding_tests;
#[cfg(test)]
mod safetensors_round_trip_io_tests;
#[cfg(test)]
mod safetensors_round_trip_tests;
#[cfg(test)]
mod safetensors_shards_tests;
#[cfg(test)]
mod safetensors_validation_tests;
#[cfg(test)]
mod safetensors_wiring_tests;
