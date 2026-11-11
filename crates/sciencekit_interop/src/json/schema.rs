//! Wire schema types for the human-readable JSON debug representation
//! (spec `json-debug-format`).

use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::io;

use sciencekit_common::{
    SKError, SKFloat, SKModelState, SKParameterValue, SKTensorDataType, SKTensorValues,
};
use serde::{Serialize, Serializer};

use super::{SK_JSON_DEBUG_FORMAT_VERSION, SKJsonFormatMode};

fn is_version_one(version: &u32) -> bool {
    *version == 1
}

/// Borrowed wire representation of [`SKModelState`] for deterministic JSON serialization.
#[derive(Debug, Serialize)]
pub(crate) struct JsonModelDocument<'state> {
    format_version: u32,
    estimator: &'state str,
    #[serde(skip_serializing_if = "is_version_one")]
    estimator_format_version: u32,
    #[serde(rename = "dtype")]
    scalar_data_type: &'static str,
    feature_count: usize,
    class_labels: Option<&'state [String]>,
    hyperparameters: BTreeMap<&'state str, JsonParameterRef<'state>>,
    tensors: BTreeMap<&'state str, JsonTensorDocument<'state>>,
}

/// Borrowed wire representation of a single named tensor.
#[derive(Debug, Serialize)]
pub(crate) struct JsonTensorDocument<'state> {
    #[serde(rename = "dtype")]
    data_type: &'static str,
    shape: &'state [usize],
    values: JsonTensorValuesRef<'state>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum JsonTensorValuesRef<'state> {
    Float32(&'state [f32]),
    Float64(&'state [f64]),
    Integer64(&'state [i64]),
    Boolean(&'state [bool]),
}

#[derive(Debug, Clone, Copy)]
struct JsonParameterRef<'state>(&'state SKParameterValue);

impl Serialize for JsonParameterRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            SKParameterValue::Bool(flag) => serializer.serialize_bool(*flag),
            SKParameterValue::Integer(integer) => serializer.serialize_i64(*integer),
            SKParameterValue::Float(float) => serializer.serialize_f64(*float),
            SKParameterValue::String(text) => serializer.serialize_str(text),
            SKParameterValue::List(items) => {
                let wrapped: Vec<JsonParameterRef<'_>> =
                    items.iter().map(JsonParameterRef).collect();
                wrapped.serialize(serializer)
            }
            _ => Err(serde::ser::Error::custom(
                "unsupported hyperparameter variant for JSON debug serialization",
            )),
        }
    }
}

/// Return the expected tensor data type and wire scalar name for `F`.
pub(crate) fn expected_scalar_descriptor<F: SKFloat>() -> (SKTensorDataType, &'static str) {
    match TypeId::of::<F>() == TypeId::of::<f32>() {
        true => (SKTensorDataType::F32, "f32"),
        false => (SKTensorDataType::F64, "f64"),
    }
}

fn validate_parameter_finiteness(value: &SKParameterValue) -> Result<(), SKError> {
    match value {
        SKParameterValue::Float(float) if !float.is_finite() => {
            Err(SKError::NonFiniteValue { position: 0 })
        }
        SKParameterValue::List(items) => items.iter().try_for_each(validate_parameter_finiteness),
        _ => Ok(()),
    }
}

fn ensure_finite<F: SKFloat>(elements: &[F]) -> Result<(), SKError> {
    elements
        .iter()
        .position(|item| !item.is_finite())
        .map_or(Ok(()), |position| Err(SKError::NonFiniteValue { position }))
}

fn borrow_tensor_values<F: SKFloat>(
    values: &SKTensorValues<F>,
) -> Result<JsonTensorValuesRef<'_>, SKError> {
    match values {
        SKTensorValues::Floating(elements) => {
            ensure_finite(elements)?;
            let any_elements = elements as &dyn Any;
            if let Some(single) = any_elements.downcast_ref::<Vec<f32>>() {
                Ok(JsonTensorValuesRef::Float32(single.as_slice()))
            } else if let Some(double) = any_elements.downcast_ref::<Vec<f64>>() {
                Ok(JsonTensorValuesRef::Float64(double.as_slice()))
            } else {
                Err(SKError::UnsupportedRepresentation {
                    representation: "unknown floating scalar in SKTensorValues",
                    suggestion: "export an f32 or f64 model state",
                })
            }
        }
        SKTensorValues::Integer(elements) => Ok(JsonTensorValuesRef::Integer64(elements)),
        SKTensorValues::Boolean(elements) => Ok(JsonTensorValuesRef::Boolean(elements)),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown tensor payload variant",
            suggestion: "export F32, F64, I64 or Bool tensors",
        }),
    }
}

/// Validate a model state up front and borrow it into a serializable document.
pub(crate) fn build_json_document<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<JsonModelDocument<'_>, SKError> {
    let (expected_data_type, scalar_data_type) = expected_scalar_descriptor::<F>();
    if state.metadata().scalar_data_type() != expected_data_type {
        return Err(SKError::InvalidHyperparameter {
            name: "dtype",
            reason: format!(
                "metadata scalar data type `{}` disagrees with model scalar `{scalar_data_type}`",
                state.metadata().scalar_data_type().as_str()
            ),
        });
    }

    let mut hyperparameters = BTreeMap::new();
    for (name, parameter) in state.metadata().parameters() {
        validate_parameter_finiteness(parameter)?;
        hyperparameters.insert(name.as_str(), JsonParameterRef(parameter));
    }

    let mut tensors = BTreeMap::new();
    for (name, tensor) in state.tensors() {
        let values = borrow_tensor_values(tensor.values())?;
        let data_type = tensor.data_type().as_str();
        let entry = JsonTensorDocument {
            data_type,
            shape: tensor.shape(),
            values,
        };
        tensors.insert(name.as_str(), entry);
    }

    Ok(JsonModelDocument {
        format_version: SK_JSON_DEBUG_FORMAT_VERSION,
        estimator: state.metadata().estimator_name(),
        estimator_format_version: state.metadata().format_version(),
        scalar_data_type,
        feature_count: state.metadata().feature_count(),
        class_labels: state.metadata().class_labels(),
        hyperparameters,
        tensors,
    })
}

/// Stream a validated model state into an arbitrary [`io::Write`] sink.
pub(crate) fn encode_model_state_to_writer<F: SKFloat, W: io::Write>(
    state: &SKModelState<F>,
    format_mode: SKJsonFormatMode,
    writer: W,
) -> Result<(), SKError> {
    let document = build_json_document(state)?;
    let result = match format_mode {
        SKJsonFormatMode::Pretty => serde_json::to_writer_pretty(writer, &document),
        SKJsonFormatMode::Compact => serde_json::to_writer(writer, &document),
    };
    result.map_err(|error| {
        if error.is_io() {
            SKError::Io(io::Error::other(error))
        } else {
            SKError::Conversion(format!("JSON serialization failed: {error}"))
        }
    })
}

/// Serialize a validated model state into a UTF-8 JSON string.
pub(crate) fn encode_model_state_to_json_string<F: SKFloat>(
    state: &SKModelState<F>,
    format_mode: SKJsonFormatMode,
) -> Result<String, SKError> {
    let mut bytes = Vec::new();
    encode_model_state_to_writer(state, format_mode, &mut bytes)?;
    String::from_utf8(bytes).map_err(|error| SKError::Conversion(error.to_string()))
}
