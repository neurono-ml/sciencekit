//! Wire-level Safetensors header, duplicate-key, offset and shape validation
//! (spec `safetensors-format`).

use std::collections::{BTreeMap, BTreeSet};

use sciencekit_common::SKError;

use super::header::{
    METADATA_FORMAT_VERSION_KEY, SK_SAFETENSORS_FORMAT_VERSION, SK_SAFETENSORS_PADDING_TENSOR_NAME,
};
use super::wire_header_deserialization::{OrderedHeaderEntries, RawHeaderValue, RawTensorHeader};

fn byte_width_for_wire_dtype(name: &str, data_type: &str) -> Result<usize, SKError> {
    if name == SK_SAFETENSORS_PADDING_TENSOR_NAME {
        return match data_type {
            "U8" => Ok(1),
            other => Err(SKError::InvalidHyperparameter {
                name: "tensor_data_type",
                reason: format!("reserved padding tensor must use `U8`, found `{other}`"),
            }),
        };
    }
    match data_type {
        "F32" => Ok(4),
        "F64" | "I64" => Ok(8),
        "BOOL" => Ok(1),
        other => Err(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            reason: format!(
                "unsupported Safetensors dtype `{other}` for tensor `{name}`; expected F32, F64, I64 or BOOL"
            ),
        }),
    }
}

fn check_future_version(metadata: &BTreeMap<String, String>) -> Result<(), SKError> {
    if let Some(raw_version) = metadata.get(METADATA_FORMAT_VERSION_KEY) {
        let major = raw_version.split('.').next().unwrap_or(raw_version);
        if let Ok(parsed) = major.parse::<u32>() {
            if parsed != SK_SAFETENSORS_FORMAT_VERSION {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "Safetensors format major version",
                    suggestion: "re-export the model state with sciencekit.format_version 1",
                });
            }
        }
    }
    Ok(())
}

fn validate_tensor_offsets(
    tensors: &[(String, RawTensorHeader)],
    payload_length: usize,
) -> Result<(), SKError> {
    let mut offset_ranges = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let [begin, end] = tensor.data_offsets;
        if begin > end {
            return Err(SKError::Conversion(format!(
                "inverted data_offsets [{begin}, {end}] for tensor `{name}`"
            )));
        }
        if end > payload_length {
            return Err(SKError::Conversion(format!(
                "truncated Safetensors payload for tensor `{name}`: end {end} exceeds {payload_length}"
            )));
        }
        let byte_width = byte_width_for_wire_dtype(name, &tensor.data_type)?;
        let declared_elements = tensor
            .shape
            .iter()
            .try_fold(1_usize, |product, &dimension| {
                product.checked_mul(dimension)
            });
        let expected_bytes = declared_elements.and_then(|count| count.checked_mul(byte_width));
        let actual_bytes = end - begin;
        if expected_bytes != Some(actual_bytes) {
            return Err(SKError::ShapeMismatch {
                expected: tensor.shape.clone(),
                found: vec![actual_bytes / byte_width],
            });
        }
        offset_ranges.push((begin, end));
    }

    offset_ranges.sort_by_key(|&(begin, _)| begin);
    let mut cursor = 0_usize;
    for (begin, end) in offset_ranges {
        if begin != cursor {
            return Err(SKError::Conversion(format!(
                "non-contiguous Safetensors offsets: expected {cursor}, found {begin}"
            )));
        }
        cursor = end;
    }
    if cursor != payload_length {
        return Err(SKError::Conversion(format!(
            "Safetensors payload length {payload_length} disagrees with tensor end {cursor}"
        )));
    }
    Ok(())
}

/// Validate header bounds, duplicate keys, container version, shapes and contiguous offsets.
pub(crate) fn validate_wire_safetensors(bytes: &[u8]) -> Result<BTreeMap<String, String>, SKError> {
    if bytes.len() < 8 {
        return Err(SKError::Conversion(
            "truncated Safetensors buffer: fewer than 8 header-length bytes".to_string(),
        ));
    }
    let header_length_bytes: [u8; 8] = bytes[..8].try_into().expect("slice length checked to be 8");
    let header_length = usize::try_from(u64::from_le_bytes(header_length_bytes)).map_err(|_| {
        SKError::Conversion("Safetensors header length overflows usize".to_string())
    })?;
    let header_end = 8_usize.checked_add(header_length).ok_or_else(|| {
        SKError::Conversion("Safetensors header end offset overflows usize".to_string())
    })?;
    if bytes.len() < header_end {
        return Err(SKError::Conversion(
            "truncated Safetensors buffer: file ends inside JSON header".to_string(),
        ));
    }

    let header_text = std::str::from_utf8(&bytes[8..header_end]).map_err(|error| {
        SKError::Conversion(format!("Safetensors header is not valid UTF-8: {error}"))
    })?;
    let OrderedHeaderEntries(entries) = serde_json::from_str(header_text).map_err(|error| {
        SKError::Conversion(format!("invalid Safetensors JSON header: {error}"))
    })?;

    let mut seen_top_keys = BTreeSet::new();
    let mut metadata_map: Option<BTreeMap<String, String>> = None;
    let mut tensors = Vec::new();
    for (key, value) in entries {
        if key.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_name",
                reason: "a tensor name in Safetensors header must not be empty".to_string(),
            });
        }
        if !seen_top_keys.insert(key.clone()) {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_names",
                reason: format!("duplicate key `{key}` in Safetensors JSON header"),
            });
        }
        match value {
            RawHeaderValue::Metadata(pairs) => {
                let mut deduplicated = BTreeMap::new();
                for (meta_key, meta_value) in pairs {
                    if deduplicated.insert(meta_key.clone(), meta_value).is_some() {
                        return Err(SKError::InvalidHyperparameter {
                            name: "metadata",
                            reason: format!("duplicate metadata key `{meta_key}`"),
                        });
                    }
                }
                metadata_map = Some(deduplicated);
            }
            RawHeaderValue::Tensor(tensor_header) => tensors.push((key, tensor_header)),
        }
    }

    let metadata = metadata_map.ok_or_else(|| {
        SKError::Conversion("Safetensors header is missing `__metadata__`".to_string())
    })?;
    check_future_version(&metadata)?;
    validate_tensor_offsets(&tensors, bytes.len() - header_end)?;
    Ok(metadata)
}
