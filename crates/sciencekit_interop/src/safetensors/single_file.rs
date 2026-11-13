//! Single-file Safetensors encoding and decoding over [`sciencekit_common::SKModelState`]
//! (spec `safetensors-format`).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use safetensors::SafeTensors;
use sciencekit_common::{SKError, SKFloat, SKModelMetadata, SKModelState, SKTensor};

use super::header::{
    SK_SAFETENSORS_PADDING_TENSOR_NAME, decode_metadata_header, encode_metadata_header,
    expected_scalar_data_type,
};
use super::padding_encode::{
    WireTensorDescriptor, serialize_safetensors_header_json, wire_dtype_label,
};
use super::shard_index::sk_safetensors_index_path;
use super::tensor_codec::{decode_tensor_payload, encode_tensor_payload};
use super::wire_validation::validate_wire_safetensors;

/// Serialize an [`SKModelState`] into standard single-file Safetensors bytes.
///
/// The header is assembled from ordered maps only: metadata keys are sorted and
/// tensor descriptors follow tensor-name order, so encoding one state twice
/// yields byte-identical artifacts.
pub(crate) fn encode_single_file_bytes<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<Vec<u8>, SKError> {
    let metadata = encode_metadata_header::<F>(state.metadata())?;

    // Match the standard container convention: descending dtype alignment,
    // then tensor name. This keeps payload offsets stable across runs while
    // preferring aligned data first.
    let mut tensors = Vec::with_capacity(state.number_of_tensors());
    for (name, tensor) in state.tensors() {
        let (dtype, shape, payload) = encode_tensor_payload(tensor)?;
        tensors.push((name.clone(), dtype, shape, payload));
    }
    tensors.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));

    let mut descriptors = Vec::with_capacity(tensors.len());
    let mut concatenated_payload = Vec::new();
    for (name, dtype, shape, payload) in tensors {
        let begin = concatenated_payload.len();
        concatenated_payload.extend_from_slice(&payload);
        descriptors.push((
            name,
            WireTensorDescriptor {
                data_type: wire_dtype_label(dtype).to_string(),
                shape,
                data_offsets: [begin, concatenated_payload.len()],
            },
        ));
    }

    let mut header_bytes =
        serialize_safetensors_header_json(&metadata, &descriptors, 0, concatenated_payload.len())?;
    let aligned_header_length = header_bytes.len().next_multiple_of(8);
    header_bytes.resize(aligned_header_length, b' ');

    let mut output = Vec::with_capacity(8 + header_bytes.len() + concatenated_payload.len());
    output.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(&header_bytes);
    output.extend_from_slice(&concatenated_payload);
    Ok(output)
}

/// Decode a single uncompressed Safetensors buffer into metadata and tensors.
pub(crate) fn decode_single_file_parts<F: SKFloat>(
    bytes: &[u8],
) -> Result<(SKModelMetadata, Vec<SKTensor<F>>), SKError> {
    let metadata_entries = validate_wire_safetensors(bytes)?;
    let metadata = decode_metadata_header::<F>(&metadata_entries)?;
    let expected_scalar = expected_scalar_data_type::<F>();

    let container = SafeTensors::deserialize(bytes)
        .map_err(|error| SKError::Conversion(format!("invalid Safetensors payload: {error}")))?;

    let mut tensors = Vec::with_capacity(container.len());
    for (name, view) in container.tensors() {
        if name == SK_SAFETENSORS_PADDING_TENSOR_NAME {
            continue;
        }
        let tensor = decode_tensor_payload::<F>(
            &name,
            view.dtype(),
            view.shape(),
            view.data(),
            expected_scalar,
        )?;
        tensors.push(tensor);
    }

    Ok((metadata, tensors))
}

/// Decode a single uncompressed Safetensors buffer into an [`SKModelState`].
pub(crate) fn decode_single_file_bytes<F: SKFloat>(
    bytes: &[u8],
) -> Result<SKModelState<F>, SKError> {
    let (metadata, tensors) = decode_single_file_parts::<F>(bytes)?;
    SKModelState::new(metadata, tensors)
}

/// Write `encoded` bytes to `path` and clear any stale companion `.index.json` manifest.
pub(crate) fn write_single_file_bytes_to_path(path: &Path, encoded: &[u8]) -> Result<(), SKError> {
    let mut buffered = BufWriter::new(File::create(path)?);
    buffered.write_all(encoded)?;
    buffered.flush()?;
    drop(buffered);

    let companion_index = sk_safetensors_index_path(path);
    if companion_index != path && companion_index.exists() {
        fs::remove_file(companion_index)?;
    }
    Ok(())
}
