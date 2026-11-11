//! Metadata header constants and codec for the extended Safetensors format
//! (spec `safetensors-format`).

/// Major format version of the extended Safetensors container written by this build.
pub const SK_SAFETENSORS_FORMAT_VERSION: u32 = 1;

/// Reserved tensor name holding zero-filled payload padding for in-place appends.
///
/// The reader excludes this entry from reconstructed [`sciencekit_common::SKModelState`]
/// tensors, and the writer rejects user tensors carrying this reserved name.
pub const SK_SAFETENSORS_PADDING_TENSOR_NAME: &str = "__padding__";
