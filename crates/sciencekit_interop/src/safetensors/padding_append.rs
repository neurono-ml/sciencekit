//! In-place padded tensor append and automatic overflow-to-new-shard fallback
//! (spec `safetensors-format`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use sciencekit_common::{SKError, SKFloat, SKModelState, SKTensor};

use super::compression::decompress_safetensors_bytes;
use super::compression_types::{SKSafetensorsCompression, sk_detect_safetensors_compression};
use super::header::{
    SK_SAFETENSORS_FORMAT_VERSION, SK_SAFETENSORS_PADDING_TENSOR_NAME, decode_metadata_header,
};
use super::padding_encode::{
    METADATA_PADDING_BYTES_KEY, SKAppendOutcome, WireTensorDescriptor,
    serialize_safetensors_header_json, validate_new_tensors, wire_dtype_label,
};
use super::shard_index::{sk_safetensors_index_path, write_shard_index_atomic};
use super::shard_io::{append_to_existing_shard_set, next_available_shard_name};
use super::shard_plan::SKShardIndex;
use super::single_file::encode_single_file_bytes;
use super::tensor_codec::encode_tensor_payload;
use super::wire_header_deserialization::{OrderedHeaderEntries, RawHeaderValue};
use super::wire_validation::validate_wire_safetensors;

/// Append `new_tensors` in place when padding suffices, otherwise spill to a new shard.
pub(crate) fn append_tensors_to_path<F: SKFloat>(
    path: &Path,
    new_tensors: &[SKTensor<F>],
    extension: &str,
    encode_shard: impl Fn(&[u8]) -> Result<Vec<u8>, SKError>,
) -> Result<SKAppendOutcome, SKError> {
    if new_tensors.is_empty() {
        return Ok(SKAppendOutcome::InPlace);
    }
    let index_path = sk_safetensors_index_path(path);
    if path.to_string_lossy().ends_with(".index.json") || index_path.exists() {
        return append_to_existing_shard_set(
            path,
            &index_path,
            new_tensors,
            extension,
            validate_new_tensors,
            encode_shard,
        );
    }

    let raw_file_bytes = fs::read(path)?;
    let is_compressed = sk_detect_safetensors_compression(&raw_file_bytes).is_some()
        || SKSafetensorsCompression::from_path(path).is_some();
    let file_bytes = decompress_safetensors_bytes(Some(path), &raw_file_bytes)?;
    let mut metadata_map = validate_wire_safetensors(&file_bytes)?;
    let parsed_metadata = decode_metadata_header::<F>(&metadata_map)?;

    let header_capacity = u64::from_le_bytes(file_bytes[..8].try_into().unwrap()) as usize;
    let header_text = std::str::from_utf8(&file_bytes[8..8 + header_capacity])
        .map_err(|error| SKError::Conversion(format!("invalid UTF-8 header: {error}")))?;
    let OrderedHeaderEntries(entries) = serde_json::from_str(header_text)
        .map_err(|error| SKError::Conversion(format!("invalid header JSON: {error}")))?;

    let mut real_descriptors = Vec::new();
    let mut existing_names = BTreeSet::new();
    let mut used_payload_end = 0_usize;
    let mut padding_span = 0_usize;

    for (name, value) in entries {
        if let RawHeaderValue::Tensor(raw) = value {
            if name == SK_SAFETENSORS_PADDING_TENSOR_NAME {
                padding_span = raw.data_offsets[1] - raw.data_offsets[0];
            } else {
                used_payload_end = used_payload_end.max(raw.data_offsets[1]);
                existing_names.insert(name.clone());
                real_descriptors.push((
                    name,
                    WireTensorDescriptor {
                        data_type: raw.data_type,
                        shape: raw.shape,
                        data_offsets: raw.data_offsets,
                    },
                ));
            }
        }
    }
    validate_new_tensors(new_tensors, &existing_names)?;

    let mut encoded_additions = Vec::with_capacity(new_tensors.len());
    let mut added_bytes = 0_usize;
    for tensor in new_tensors {
        let (dtype, shape, payload) = encode_tensor_payload(tensor)?;
        added_bytes += payload.len();
        encoded_additions.push((tensor.name().to_string(), dtype, shape, payload));
    }

    if !is_compressed && added_bytes <= padding_span {
        let remaining_padding = padding_span - added_bytes;
        let mut updated_metadata = metadata_map.clone();
        updated_metadata.insert(
            METADATA_PADDING_BYTES_KEY.to_string(),
            remaining_padding.to_string(),
        );
        let mut candidate_descriptors = real_descriptors.clone();
        let mut cursor = used_payload_end;
        for (name, dtype, shape, payload) in &encoded_additions {
            let next = cursor + payload.len();
            candidate_descriptors.push((
                name.clone(),
                WireTensorDescriptor {
                    data_type: wire_dtype_label(*dtype).to_string(),
                    shape: shape.clone(),
                    data_offsets: [cursor, next],
                },
            ));
            cursor = next;
        }
        let mut candidate_header = serialize_safetensors_header_json(
            &updated_metadata,
            &candidate_descriptors,
            remaining_padding,
            cursor,
        )?;
        if candidate_header.len() <= header_capacity {
            candidate_header.resize(header_capacity, b' ');
            let mut file = OpenOptions::new().read(true).write(true).open(path)?;
            file.seek(SeekFrom::Start(8))?;
            file.write_all(&candidate_header)?;
            file.seek(SeekFrom::Start(
                (8 + header_capacity + used_payload_end) as u64,
            ))?;
            for (_name, _dtype, _shape, payload) in &encoded_additions {
                file.write_all(payload)?;
            }
            file.flush()?;
            return Ok(SKAppendOutcome::InPlace);
        }
    }

    let base_name = path
        .file_name()
        .and_then(|os| os.to_str())
        .ok_or_else(|| SKError::Conversion("invalid base Safetensors filename".to_string()))?
        .to_string();
    let shard_file = next_available_shard_name(path, std::slice::from_ref(&base_name), extension);
    let shard_state = SKModelState::new(parsed_metadata, new_tensors.to_vec())?;
    let encoded_shard = encode_shard(&encode_single_file_bytes(&shard_state)?)?;
    fs::write(
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&shard_file),
        encoded_shard,
    )?;

    metadata_map.remove(METADATA_PADDING_BYTES_KEY);
    metadata_map.insert(
        "total_size".to_string(),
        (used_payload_end + added_bytes).to_string(),
    );
    let tensor_to_shard: BTreeMap<String, String> = real_descriptors
        .iter()
        .map(|(name, _)| (name.clone(), base_name.clone()))
        .chain(
            new_tensors
                .iter()
                .map(|tensor| (tensor.name().to_string(), shard_file.clone())),
        )
        .collect();
    let index = SKShardIndex {
        format_version: SK_SAFETENSORS_FORMAT_VERSION,
        metadata: metadata_map,
        shards: vec![base_name, shard_file.clone()],
        tensor_to_shard,
    };
    write_shard_index_atomic(&index_path, &index)?;
    Ok(SKAppendOutcome::NewShard { shard_file })
}
