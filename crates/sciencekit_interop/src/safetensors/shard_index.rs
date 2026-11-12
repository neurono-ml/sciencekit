//! Shard-index manifest paths, atomic persistence and validation
//! (spec `safetensors-format`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use sciencekit_common::SKError;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize};

use super::header::SK_SAFETENSORS_FORMAT_VERSION;
use super::shard_plan::SKShardIndex;

static ATOMIC_TEMPORARY_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Serialize)]
struct WireShardIndex<'index> {
    format_version: u32,
    metadata: &'index BTreeMap<String, String>,
    shards: &'index [String],
    weight_map: &'index BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RawIndexVersionEnvelope {
    format_version: u32,
}

#[derive(Deserialize)]
struct RawShardIndex {
    format_version: u32,
    metadata: BTreeMap<String, String>,
    shards: Vec<String>,
    #[serde(deserialize_with = "deserialize_ordered_string_pairs")]
    weight_map: Vec<(String, String)>,
}

fn deserialize_ordered_string_pairs<'de, D>(
    deserializer: D,
) -> Result<Vec<(String, String)>, D::Error>
where
    D: Deserializer<'de>,
{
    struct PairVisitor;

    impl<'de> Visitor<'de> for PairVisitor {
        type Value = Vec<(String, String)>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a JSON tensor-to-shard map")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut pairs = Vec::new();
            while let Some(pair) = map.next_entry::<String, String>()? {
                pairs.push(pair);
            }
            Ok(pairs)
        }
    }

    deserializer.deserialize_map(PairVisitor)
}

/// Resolve the `.index.json` manifest path corresponding to `path`.
pub fn sk_safetensors_index_path(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text.ends_with(".index.json") {
        return path.to_path_buf();
    }
    let mut name = path
        .file_name()
        .map(|operating_system_name| operating_system_name.to_os_string())
        .unwrap_or_default();
    name.push(".index.json");
    path.with_file_name(name)
}

pub(crate) fn shard_file_name_for_number(
    base_path: &Path,
    shard_number: usize,
    extension: &str,
) -> String {
    let raw_name = base_path
        .file_name()
        .map(|operating_system_name| operating_system_name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let stem = raw_name
        .strip_suffix(".index.json")
        .or_else(|| raw_name.strip_suffix(".safetensors.gz"))
        .or_else(|| raw_name.strip_suffix(".safetensors.brotli"))
        .or_else(|| raw_name.strip_suffix(".snappy.safetensors"))
        .or_else(|| raw_name.strip_suffix(".safetensors"))
        .unwrap_or(&raw_name);
    format!("{stem}.shard-{shard_number:05}.{extension}")
}

fn validate_relative_shard_name(shard_name: &str) -> Result<(), SKError> {
    if shard_name.is_empty() {
        return Err(SKError::InvalidHyperparameter {
            name: "shards",
            reason: "shard filename must not be empty".to_string(),
        });
    }
    let mut components = Path::new(shard_name).components();
    let valid =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !valid {
        return Err(SKError::InvalidHyperparameter {
            name: "shards",
            reason: format!("shard filename `{shard_name}` must be a plain relative filename"),
        });
    }
    Ok(())
}

/// Atomically serialize and rename `index` onto `index_path`.
pub(crate) fn write_shard_index_atomic(
    index_path: &Path,
    index: &SKShardIndex,
) -> Result<(), SKError> {
    let parent = index_path.parent().unwrap_or_else(|| Path::new("."));
    let sequence = ATOMIC_TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary_name = format!(".safetensors-index-{}-{sequence}.tmp", std::process::id());
    let temporary_path = parent.join(temporary_name);

    let wire = WireShardIndex {
        format_version: index.format_version,
        metadata: &index.metadata,
        shards: &index.shards,
        weight_map: &index.tensor_to_shard,
    };
    let file = File::create(&temporary_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &wire).map_err(|error| {
        SKError::Conversion(format!("failed to serialize shard index: {error}"))
    })?;
    writer.flush()?;
    drop(writer);
    fs::rename(&temporary_path, index_path)?;
    Ok(())
}

/// Parse and validate a `.safetensors.index.json` manifest at `index_path`.
pub fn sk_read_safetensors_shard_index(index_path: &Path) -> Result<SKShardIndex, SKError> {
    let raw_text = fs::read_to_string(index_path)?;
    let envelope: RawIndexVersionEnvelope = serde_json::from_str(&raw_text)
        .map_err(|error| SKError::Conversion(format!("invalid shard index JSON: {error}")))?;
    if envelope.format_version != SK_SAFETENSORS_FORMAT_VERSION {
        return Err(SKError::UnsupportedRepresentation {
            representation: "Safetensors shard index major version",
            suggestion: "re-export the sharded model state with format_version 1",
        });
    }
    let raw: RawShardIndex = serde_json::from_str(&raw_text)
        .map_err(|error| SKError::Conversion(format!("invalid shard index JSON: {error}")))?;
    if raw.shards.is_empty() {
        return Err(SKError::EmptyInput {
            operation: "read_safetensors_shards",
        });
    }
    let mut seen_shards = BTreeSet::new();
    for shard in &raw.shards {
        validate_relative_shard_name(shard)?;
        if !seen_shards.insert(shard.as_str()) {
            return Err(SKError::InvalidHyperparameter {
                name: "shards",
                reason: format!("duplicate shard `{shard}` in shard index"),
            });
        }
    }
    let mut tensor_to_shard = BTreeMap::new();
    for (tensor_name, shard_name) in raw.weight_map {
        if !seen_shards.contains(shard_name.as_str()) {
            return Err(SKError::InvalidHyperparameter {
                name: "weight_map",
                reason: format!("tensor `{tensor_name}` references unlisted shard `{shard_name}`"),
            });
        }
        if tensor_to_shard
            .insert(tensor_name.clone(), shard_name)
            .is_some()
        {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_names",
                reason: format!("duplicate tensor `{tensor_name}` in shard index"),
            });
        }
    }
    Ok(SKShardIndex {
        format_version: raw.format_version,
        metadata: raw.metadata,
        shards: raw.shards,
        tensor_to_shard,
    })
}
