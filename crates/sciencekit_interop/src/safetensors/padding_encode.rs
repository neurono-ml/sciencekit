//! Header and payload padding encoder for in-place Safetensors appends
//! (spec `safetensors-format`).

use std::collections::{BTreeMap, BTreeSet};

use safetensors::Dtype;
use sciencekit_common::{SKError, SKFloat, SKModelState, SKTensor};
use serde::Serialize;
use serde::ser::SerializeMap;

use super::header::{
    SK_SAFETENSORS_PADDING_TENSOR_NAME, encode_metadata_header, expected_scalar_data_type,
};
use super::single_file::encode_single_file_bytes;
use super::tensor_codec::encode_tensor_payload;

pub(crate) const METADATA_PADDING_BYTES_KEY: &str = "__padding_bytes__";

/// Outcome of appending tensor(s) to an existing Safetensors path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SKAppendOutcome {
    /// The new tensor(s) fit inside reserved header and payload padding and were written in place.
    InPlace,
    /// Reserved padding was exhausted or absent, so an immutable shard was created and indexed.
    NewShard {
        /// Relative filename of the newly written shard.
        shard_file: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireTensorDescriptor {
    #[serde(rename = "dtype")]
    pub(crate) data_type: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: [usize; 2],
}

pub(crate) fn wire_dtype_label(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "F32",
        Dtype::F64 => "F64",
        Dtype::I64 => "I64",
        Dtype::BOOL => "BOOL",
        Dtype::U8 => "U8",
        _ => "U8",
    }
}

struct OrderedSafetensorsHeader<'header> {
    metadata: &'header BTreeMap<String, String>,
    tensors: &'header [(String, WireTensorDescriptor)],
}

impl Serialize for OrderedSafetensorsHeader<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(1 + self.tensors.len()))?;
        map.serialize_entry("__metadata__", self.metadata)?;
        for (name, descriptor) in self.tensors {
            map.serialize_entry(name, descriptor)?;
        }
        map.end()
    }
}

/// Serialize a Safetensors JSON header object with `__metadata__` first and optional `__padding__`.
pub(crate) fn serialize_safetensors_header_json(
    metadata: &BTreeMap<String, String>,
    real_tensors: &[(String, WireTensorDescriptor)],
    remaining_payload_padding: usize,
    used_payload_end: usize,
) -> Result<Vec<u8>, SKError> {
    let mut all_tensors = real_tensors.to_vec();
    if remaining_payload_padding > 0 {
        all_tensors.push((
            SK_SAFETENSORS_PADDING_TENSOR_NAME.to_string(),
            WireTensorDescriptor {
                data_type: "U8".to_string(),
                shape: vec![remaining_payload_padding],
                data_offsets: [
                    used_payload_end,
                    used_payload_end + remaining_payload_padding,
                ],
            },
        ));
    }
    let header = OrderedSafetensorsHeader {
        metadata,
        tensors: &all_tensors,
    };
    serde_json::to_vec(&header).map_err(|error| {
        SKError::Conversion(format!("failed to serialize Safetensors header: {error}"))
    })
}

/// Encode `state` with reserved header space (`0x20` padding) and zero-filled payload padding.
pub(crate) fn encode_state_with_padding<F: SKFloat>(
    state: &SKModelState<F>,
    header_padding_bytes: usize,
    payload_padding_bytes: usize,
) -> Result<Vec<u8>, SKError> {
    if header_padding_bytes == 0 && payload_padding_bytes == 0 {
        return encode_single_file_bytes(state);
    }

    let mut metadata = encode_metadata_header::<F>(state.metadata())?;
    metadata.insert(
        METADATA_PADDING_BYTES_KEY.to_string(),
        payload_padding_bytes.to_string(),
    );

    let mut descriptors = Vec::with_capacity(state.number_of_tensors());
    let mut concatenated_payload = Vec::new();
    for (name, tensor) in state.tensors() {
        let (dtype, shape, bytes) = encode_tensor_payload(tensor)?;
        let begin = concatenated_payload.len();
        concatenated_payload.extend_from_slice(&bytes);
        let end = concatenated_payload.len();
        descriptors.push((
            name.clone(),
            WireTensorDescriptor {
                data_type: wire_dtype_label(dtype).to_string(),
                shape,
                data_offsets: [begin, end],
            },
        ));
    }

    let used_payload_end = concatenated_payload.len();
    let mut header_bytes = serialize_safetensors_header_json(
        &metadata,
        &descriptors,
        payload_padding_bytes,
        used_payload_end,
    )?;
    let target_header_capacity = header_bytes
        .len()
        .saturating_add(header_padding_bytes)
        .next_multiple_of(8);
    header_bytes.resize(target_header_capacity, b' ');

    let total_size = 8 + header_bytes.len() + used_payload_end + payload_padding_bytes;
    let mut output = Vec::with_capacity(total_size);
    output.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(&header_bytes);
    output.extend_from_slice(&concatenated_payload);
    output.resize(total_size, 0_u8);
    Ok(output)
}

/// Validate a slice of tensors being appended onto an existing state.
pub(crate) fn validate_new_tensors<F: SKFloat>(
    new_tensors: &[SKTensor<F>],
    existing_names: &BTreeSet<String>,
) -> Result<(), SKError> {
    let expected_scalar = expected_scalar_data_type::<F>();
    let mut batch_names = BTreeSet::new();
    for tensor in new_tensors {
        if tensor.name() == SK_SAFETENSORS_PADDING_TENSOR_NAME {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_name",
                reason: "cannot append a tensor with the reserved padding name".to_string(),
            });
        }
        if tensor.data_type().is_floating() && tensor.data_type() != expected_scalar {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_data_type",
                reason: format!(
                    "tensor `{}` uses {} but target scalar is {}",
                    tensor.name(),
                    tensor.data_type().as_str(),
                    expected_scalar.as_str()
                ),
            });
        }
        if existing_names.contains(tensor.name()) || !batch_names.insert(tensor.name()) {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_names",
                reason: format!("tensor `{}` already exists in model state", tensor.name()),
            });
        }
    }
    Ok(())
}
