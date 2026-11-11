//! Convenience free-scope entry points for writing and appending Safetensors states
//! (spec `safetensors-format`).

use std::path::Path;

use sciencekit_common::{SKError, SKExportable, SKFloat, SKModelState, SKTensor};

use super::padding_encode::SKAppendOutcome;
use super::writer::SKSafetensorsWriter;

/// Serialize `state` into standard Safetensors bytes with default configuration.
pub fn sk_write_model_state_safetensors<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<Vec<u8>, SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .write_state_to_bytes(state)
}

/// Write `state` as a Safetensors file to `path` with default configuration.
pub fn sk_write_model_state_safetensors_file<F: SKFloat>(
    path: &Path,
    state: &SKModelState<F>,
) -> Result<(), SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .write_state_to_path(state, path)
}

/// Export `model` via [`SKExportable`] into standard Safetensors bytes.
pub fn sk_export_model_safetensors<F: SKFloat, M: SKExportable<F>>(
    model: &M,
) -> Result<Vec<u8>, SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .export_to_bytes(model)
}

/// Export `model` via [`SKExportable`] into a Safetensors file at `path`.
pub fn sk_export_model_safetensors_file<F: SKFloat, M: SKExportable<F>>(
    path: &Path,
    model: &M,
) -> Result<(), SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .export_to_path(model, path)
}

/// Append `tensor` in place to `path` when reserved padding suffices, or spill to a new shard.
pub fn sk_append_safetensors_tensor<F: SKFloat>(
    path: &Path,
    tensor: &SKTensor<F>,
) -> Result<SKAppendOutcome, SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .append_tensor_to_path(path, tensor)
}
