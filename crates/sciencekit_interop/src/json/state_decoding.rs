//! Validation and conversion of raw JSON documents into [`sciencekit_common::SKModelState`]
//! (spec `json-debug-format`).

use std::collections::BTreeSet;

use sciencekit_common::{
    SKError, SKFloat, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType,
};

use super::SK_JSON_DEBUG_FORMAT_VERSION;
use super::schema::expected_scalar_descriptor;
use super::schema_deserialization::{RawJsonModelDocument, RawJsonVersionEnvelope};
use super::tensor_decoding::{decode_tensor_values, validate_tensor_headers};

fn parse_model_scalar_data_type(text: &str) -> Result<SKTensorDataType, SKError> {
    match text {
        "f32" => Ok(SKTensorDataType::F32),
        "f64" => Ok(SKTensorDataType::F64),
        other => Err(SKError::InvalidHyperparameter {
            name: "dtype",
            reason: format!("unsupported model scalar dtype `{other}`; expected `f32` or `f64`"),
        }),
    }
}

/// Parse UTF-8 JSON text, checking `format_version` before typing version-1 fields.
pub(crate) fn decode_json_str<F: SKFloat>(input: &str) -> Result<SKModelState<F>, SKError> {
    let envelope: RawJsonVersionEnvelope = serde_json::from_str(input)
        .map_err(|error| SKError::Conversion(format!("invalid JSON model state: {error}")))?;
    if envelope.format_version != SK_JSON_DEBUG_FORMAT_VERSION {
        return Err(SKError::UnsupportedRepresentation {
            representation: "JSON debug format major version",
            suggestion: "re-export the model state with format_version 1",
        });
    }
    let raw: RawJsonModelDocument = serde_json::from_str(input)
        .map_err(|error| SKError::Conversion(format!("invalid JSON model state: {error}")))?;
    decode_raw_document(raw)
}

/// Validate a parsed JSON debug document up front and rebuild an [`SKModelState`].
pub(crate) fn decode_raw_document<F: SKFloat>(
    raw: RawJsonModelDocument,
) -> Result<SKModelState<F>, SKError> {
    if raw.format_version != SK_JSON_DEBUG_FORMAT_VERSION {
        return Err(SKError::UnsupportedRepresentation {
            representation: "JSON debug format major version",
            suggestion: "re-export the model state with format_version 1",
        });
    }
    let scalar_data_type = parse_model_scalar_data_type(&raw.scalar_data_type)?;
    let (expected_scalar_type, expected_name) = expected_scalar_descriptor::<F>();
    if scalar_data_type != expected_scalar_type {
        return Err(SKError::InvalidHyperparameter {
            name: "dtype",
            reason: format!(
                "document dtype `{}` disagrees with target scalar `{expected_name}`",
                raw.scalar_data_type
            ),
        });
    }

    let mut metadata = SKModelMetadata::new(
        raw.estimator,
        raw.estimator_format_version,
        scalar_data_type,
        raw.feature_count,
    )?;
    if let Some(labels) = raw.class_labels {
        metadata = metadata.with_class_labels(labels)?;
    }
    let mut seen_parameters = BTreeSet::new();
    for (name, raw_parameter) in raw.hyperparameters {
        if !seen_parameters.insert(name.clone()) {
            return Err(SKError::InvalidHyperparameter {
                name: "hyperparameters",
                reason: format!("hyperparameter `{name}` is duplicated"),
            });
        }
        metadata = metadata.with_parameter(name, raw_parameter.into_parameter_value()?)?;
    }

    let parsed_types = validate_tensor_headers(&raw.tensors, scalar_data_type)?;
    let mut tensors = Vec::with_capacity(raw.tensors.len());
    for ((name, raw_tensor), data_type) in raw.tensors.into_iter().zip(parsed_types) {
        let values = decode_tensor_values::<F>(&name, data_type, raw_tensor.values)?;
        tensors.push(SKTensor::new(name, data_type, raw_tensor.shape, values)?);
    }

    SKModelState::new(metadata, tensors)
}
