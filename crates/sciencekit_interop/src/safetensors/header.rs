//! Metadata header constants and codec for the extended Safetensors format
//! (spec `safetensors-format`).

use std::any::TypeId;
use std::collections::BTreeMap;

use sciencekit_common::{SKError, SKFloat, SKModelMetadata, SKTensorDataType};

use super::parameter_codec::{decode_parameters_json, encode_parameters_json};

/// Major format version of the extended Safetensors container written by this build.
pub const SK_SAFETENSORS_FORMAT_VERSION: u32 = 1;

/// Reserved tensor name holding zero-filled payload padding for in-place appends.
///
/// The reader excludes this entry from reconstructed [`sciencekit_common::SKModelState`]
/// tensors, and the writer rejects user tensors carrying this reserved name.
pub const SK_SAFETENSORS_PADDING_TENSOR_NAME: &str = "__padding__";

pub(crate) const METADATA_FORMAT_VERSION_KEY: &str = "sciencekit.format_version";
pub(crate) const METADATA_ESTIMATOR_NAME_KEY: &str = "sciencekit.estimator_name";
pub(crate) const METADATA_ESTIMATOR_VERSION_KEY: &str = "sciencekit.estimator_format_version";
pub(crate) const METADATA_SCALAR_DATA_TYPE_KEY: &str = "sciencekit.scalar_data_type";
pub(crate) const METADATA_FEATURE_COUNT_KEY: &str = "sciencekit.feature_count";
pub(crate) const METADATA_CLASS_LABELS_KEY: &str = "sciencekit.class_labels";
pub(crate) const METADATA_PARAMETERS_KEY: &str = "sciencekit.parameters";

/// Return the [`SKTensorDataType`] matching the target Rust scalar `F`.
pub(crate) fn expected_scalar_data_type<F: SKFloat>() -> SKTensorDataType {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        SKTensorDataType::F32
    } else {
        SKTensorDataType::F64
    }
}

fn parse_scalar_data_type(value: &str) -> Result<SKTensorDataType, SKError> {
    match value {
        "F32" | "f32" => Ok(SKTensorDataType::F32),
        "F64" | "f64" => Ok(SKTensorDataType::F64),
        other => Err(SKError::InvalidHyperparameter {
            name: "scalar_data_type",
            reason: format!(
                "unsupported model scalar data type `{other}`; expected `F32` or `F64`"
            ),
        }),
    }
}

fn parse_container_version(raw_version: &str) -> Result<u32, SKError> {
    let major_prefix = raw_version.split('.').next().unwrap_or(raw_version);
    let parsed = major_prefix
        .parse::<u32>()
        .map_err(|_| SKError::InvalidHyperparameter {
            name: "format_version",
            reason: format!("invalid `sciencekit.format_version` value `{raw_version}`"),
        })?;
    if parsed != SK_SAFETENSORS_FORMAT_VERSION {
        return Err(SKError::UnsupportedRepresentation {
            representation: "Safetensors format major version",
            suggestion: "re-export the model state with sciencekit.format_version 1",
        });
    }
    Ok(parsed)
}

/// Encode [`SKModelMetadata`] into deterministic Safetensors `__metadata__` string pairs.
pub(crate) fn encode_metadata_header<F: SKFloat>(
    metadata: &SKModelMetadata,
) -> Result<BTreeMap<String, String>, SKError> {
    let expected_scalar = expected_scalar_data_type::<F>();
    if metadata.scalar_data_type() != expected_scalar {
        return Err(SKError::InvalidHyperparameter {
            name: "scalar_data_type",
            reason: format!(
                "metadata scalar data type `{}` disagrees with target scalar `{}`",
                metadata.scalar_data_type().as_str(),
                expected_scalar.as_str()
            ),
        });
    }

    let mut entries = BTreeMap::new();
    entries.insert(
        METADATA_FORMAT_VERSION_KEY.to_string(),
        SK_SAFETENSORS_FORMAT_VERSION.to_string(),
    );
    entries.insert(
        METADATA_ESTIMATOR_NAME_KEY.to_string(),
        metadata.estimator_name().to_string(),
    );
    entries.insert(
        METADATA_ESTIMATOR_VERSION_KEY.to_string(),
        metadata.format_version().to_string(),
    );
    entries.insert(
        METADATA_SCALAR_DATA_TYPE_KEY.to_string(),
        metadata.scalar_data_type().as_str().to_string(),
    );
    entries.insert(
        METADATA_FEATURE_COUNT_KEY.to_string(),
        metadata.feature_count().to_string(),
    );
    if let Some(labels) = metadata.class_labels() {
        let encoded_labels = serde_json::to_string(labels).map_err(|error| {
            SKError::Conversion(format!("failed to encode class labels: {error}"))
        })?;
        entries.insert(METADATA_CLASS_LABELS_KEY.to_string(), encoded_labels);
    }
    let encoded_parameters = encode_parameters_json(metadata.parameters())?;
    entries.insert(METADATA_PARAMETERS_KEY.to_string(), encoded_parameters);
    Ok(entries)
}

fn require_field<'map>(
    entries: &'map BTreeMap<String, String>,
    key: &'static str,
) -> Result<&'map str, SKError> {
    entries
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| SKError::Conversion(format!("missing Safetensors metadata field `{key}`")))
}

/// Decode [`SKModelMetadata`] from Safetensors `__metadata__`, ignoring unknown keys.
pub(crate) fn decode_metadata_header<F: SKFloat>(
    entries: &BTreeMap<String, String>,
) -> Result<SKModelMetadata, SKError> {
    let raw_version = require_field(entries, METADATA_FORMAT_VERSION_KEY)?;
    let _verified = parse_container_version(raw_version)?;

    let estimator_name = require_field(entries, METADATA_ESTIMATOR_NAME_KEY)?;
    let estimator_version = match entries.get(METADATA_ESTIMATOR_VERSION_KEY) {
        Some(text) => text
            .parse::<u32>()
            .map_err(|_| SKError::InvalidHyperparameter {
                name: "estimator_format_version",
                reason: format!("invalid estimator format version `{text}`"),
            })?,
        None => 1,
    };
    let scalar_data_type =
        parse_scalar_data_type(require_field(entries, METADATA_SCALAR_DATA_TYPE_KEY)?)?;
    let expected_scalar = expected_scalar_data_type::<F>();
    if scalar_data_type != expected_scalar {
        return Err(SKError::InvalidHyperparameter {
            name: "scalar_data_type",
            reason: format!(
                "file scalar data type `{}` disagrees with target scalar `{}`",
                scalar_data_type.as_str(),
                expected_scalar.as_str()
            ),
        });
    }
    let feature_count_text = require_field(entries, METADATA_FEATURE_COUNT_KEY)?;
    let feature_count =
        feature_count_text
            .parse::<usize>()
            .map_err(|_| SKError::InvalidHyperparameter {
                name: "feature_count",
                reason: format!("invalid feature count `{feature_count_text}`"),
            })?;

    let mut metadata = SKModelMetadata::new(
        estimator_name,
        estimator_version,
        scalar_data_type,
        feature_count,
    )?;
    if let Some(raw_labels) = entries.get(METADATA_CLASS_LABELS_KEY) {
        let labels: Vec<String> = serde_json::from_str(raw_labels).map_err(|error| {
            SKError::Conversion(format!(
                "invalid JSON in `sciencekit.class_labels`: {error}"
            ))
        })?;
        metadata = metadata.with_class_labels(labels)?;
    }
    if let Some(raw_parameters) = entries.get(METADATA_PARAMETERS_KEY) {
        for (name, parameter_value) in decode_parameters_json(raw_parameters)? {
            metadata = metadata.with_parameter(name, parameter_value)?;
        }
    }
    Ok(metadata)
}
