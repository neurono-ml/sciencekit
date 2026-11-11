//! Single-file Safetensors encoding and decoding over [`sciencekit_common::SKModelState`]
//! (spec `safetensors-format`).

use std::collections::{BTreeMap, HashMap};

use safetensors::SafeTensors;
use safetensors::tensor::{TensorView, serialize};
use sciencekit_common::{SKError, SKFloat, SKModelMetadata, SKModelState, SKTensor};

use super::header::{
    SK_SAFETENSORS_PADDING_TENSOR_NAME, decode_metadata_header, encode_metadata_header,
    expected_scalar_data_type,
};
use super::tensor_codec::{decode_tensor_payload, encode_tensor_payload};

/// Serialize an [`SKModelState`] into standard single-file Safetensors bytes.
pub(crate) fn encode_single_file_bytes<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<Vec<u8>, SKError> {
    let metadata_btree = encode_metadata_header::<F>(state.metadata())?;
    let metadata_map: HashMap<String, String> = metadata_btree.into_iter().collect();

    let mut owned_buffers = Vec::with_capacity(state.number_of_tensors());
    for (name, tensor) in state.tensors() {
        let (dtype, shape, payload) = encode_tensor_payload(tensor)?;
        owned_buffers.push((name.as_str(), dtype, shape, payload));
    }

    let mut views = Vec::with_capacity(owned_buffers.len());
    for (name, dtype, shape, payload) in &owned_buffers {
        let view = TensorView::new(*dtype, shape.clone(), payload).map_err(|error| {
            SKError::Conversion(format!(
                "failed to construct Safetensors view for `{name}`: {error}"
            ))
        })?;
        views.push((*name, view));
    }

    serialize(views, Some(metadata_map))
        .map_err(|error| SKError::Conversion(format!("Safetensors serialization failed: {error}")))
}

/// Decode a single uncompressed Safetensors buffer into metadata and tensors.
pub(crate) fn decode_single_file_parts<F: SKFloat>(
    bytes: &[u8],
) -> Result<(SKModelMetadata, Vec<SKTensor<F>>), SKError> {
    let (_header_size, raw_metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| SKError::Conversion(format!("invalid Safetensors header: {error}")))?;
    let metadata_entries: BTreeMap<String, String> = raw_metadata
        .metadata()
        .as_ref()
        .map(|map| {
            map.iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .ok_or_else(|| {
            SKError::Conversion("Safetensors header is missing `__metadata__`".to_string())
        })?;

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
