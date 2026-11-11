//! Shard file writing and sharded state merging (spec `safetensors-format`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use sciencekit_common::{SKError, SKFloat, SKModelState, SKTensor};

use super::header::{
    SK_SAFETENSORS_FORMAT_VERSION, decode_metadata_header, encode_metadata_header,
};
use super::shard_index::{
    shard_file_name_for_number, sk_read_safetensors_shard_index, sk_safetensors_index_path,
    write_shard_index_atomic,
};
use super::shard_plan::{SKShardIndex, SKShardPlan, state_total_payload_bytes};
use super::single_file::{decode_single_file_parts, encode_single_file_bytes};

/// Split `state` according to `plan`, writing immutable shards and an atomic index.
pub(crate) fn write_sharded_state_to_path<F: SKFloat>(
    state: &SKModelState<F>,
    base_path: &Path,
    plan: &SKShardPlan,
    extension: &str,
    encode_bytes: impl Fn(&[u8]) -> Result<Vec<u8>, SKError>,
) -> Result<(), SKError> {
    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));
    let groups = plan.partition_state(state);
    let mut metadata_map = encode_metadata_header::<F>(state.metadata())?;
    metadata_map.insert(
        "total_size".to_string(),
        state_total_payload_bytes(state).to_string(),
    );

    let mut shards = Vec::with_capacity(groups.len());
    let mut tensor_to_shard = BTreeMap::new();

    for (zero_based, group) in groups.into_iter().enumerate() {
        let shard_name = shard_file_name_for_number(base_path, zero_based + 1, extension);
        let owned_group: Vec<SKTensor<F>> = group.into_iter().cloned().collect();
        for tensor in &owned_group {
            tensor_to_shard.insert(tensor.name().to_string(), shard_name.clone());
        }
        let shard_state = SKModelState::new(state.metadata().clone(), owned_group)?;
        let raw_shard = encode_single_file_bytes(&shard_state)?;
        let final_bytes = encode_bytes(&raw_shard)?;
        fs::write(parent.join(&shard_name), final_bytes)?;
        shards.push(shard_name);
    }

    let index = SKShardIndex {
        format_version: SK_SAFETENSORS_FORMAT_VERSION,
        metadata: metadata_map,
        shards,
        tensor_to_shard,
    };
    write_shard_index_atomic(&sk_safetensors_index_path(base_path), &index)
}

/// Read every shard listed in `index_path` and merge them into one [`SKModelState`].
pub(crate) fn read_sharded_state_from_index<F: SKFloat>(
    index_path: &Path,
    decode_bytes: impl Fn(&Path, &[u8]) -> Result<Vec<u8>, SKError>,
) -> Result<SKModelState<F>, SKError> {
    let index = sk_read_safetensors_shard_index(index_path)?;
    let metadata = decode_metadata_header::<F>(&index.metadata)?;
    let parent = index_path.parent().unwrap_or_else(|| Path::new("."));

    let mut merged_tensors = Vec::with_capacity(index.tensor_to_shard.len());
    let mut seen_tensors = BTreeSet::new();
    for shard_name in &index.shards {
        let shard_path = parent.join(shard_name);
        let raw_bytes = fs::read(&shard_path)?;
        let uncompressed = decode_bytes(&shard_path, &raw_bytes)?;
        let (_shard_meta, shard_tensors) = decode_single_file_parts::<F>(&uncompressed)?;
        for tensor in shard_tensors {
            let expected_shard = index.tensor_to_shard.get(tensor.name());
            if expected_shard.map(String::as_str) != Some(shard_name.as_str()) {
                return Err(SKError::InvalidHyperparameter {
                    name: "weight_map",
                    reason: format!(
                        "shard `{shard_name}` carries tensor `{}` not mapped to it in index",
                        tensor.name()
                    ),
                });
            }
            if !seen_tensors.insert(tensor.name().to_string()) {
                return Err(SKError::InvalidHyperparameter {
                    name: "tensor_names",
                    reason: format!("duplicate tensor `{}` across shards", tensor.name()),
                });
            }
            merged_tensors.push(tensor);
        }
    }
    if seen_tensors.len() != index.tensor_to_shard.len() {
        return Err(SKError::InvalidHyperparameter {
            name: "weight_map",
            reason: "shard index declares tensors missing from the shard files".to_string(),
        });
    }
    SKModelState::new(metadata, merged_tensors)
}
