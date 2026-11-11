//! Tensor header validation and flat payload decoding for the JSON debug reader
//! (spec `json-debug-format`).

use std::any::Any;
use std::collections::BTreeSet;

use sciencekit_common::{SKError, SKFloat, SKTensorDataType, SKTensorValues};

use super::schema_deserialization::{RawJsonScalar, RawJsonTensor};

pub(crate) fn parse_tensor_data_type(text: &str) -> Result<SKTensorDataType, SKError> {
    match text {
        "F32" => Ok(SKTensorDataType::F32),
        "F64" => Ok(SKTensorDataType::F64),
        "I64" => Ok(SKTensorDataType::I64),
        "Bool" => Ok(SKTensorDataType::Bool),
        other => Err(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            reason: format!(
                "unsupported tensor dtype `{other}`; expected `F32`, `F64`, `I64` or `Bool`"
            ),
        }),
    }
}

pub(crate) fn validate_tensor_headers(
    tensors: &[(String, RawJsonTensor)],
    scalar_data_type: SKTensorDataType,
) -> Result<Vec<SKTensorDataType>, SKError> {
    let mut seen_names = BTreeSet::new();
    let mut parsed_types = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        if name.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_name",
                reason: "a tensor name must not be empty".to_string(),
            });
        }
        if !seen_names.insert(name.as_str()) {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_names",
                reason: format!("tensor name `{name}` is not unique"),
            });
        }
        let data_type = parse_tensor_data_type(&tensor.data_type)?;
        if data_type.is_floating() && data_type != scalar_data_type {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_data_type",
                reason: format!(
                    "tensor `{name}` uses {} but the model scalar is {}",
                    data_type.as_str(),
                    scalar_data_type.as_str()
                ),
            });
        }
        let declared = tensor
            .shape
            .iter()
            .try_fold(1_usize, |product, &dimension| {
                product.checked_mul(dimension)
            });
        if declared != Some(tensor.values.len()) {
            return Err(SKError::ShapeMismatch {
                expected: tensor.shape.clone(),
                found: vec![tensor.values.len()],
            });
        }
        parsed_types.push(data_type);
    }
    Ok(parsed_types)
}

fn cast_floating_vector<Source: 'static, F: SKFloat>(
    values: Vec<Source>,
) -> Result<Vec<F>, SKError> {
    let mut slot = Some(values);
    let any_slot = &mut slot as &mut dyn Any;
    any_slot
        .downcast_mut::<Option<Vec<F>>>()
        .and_then(Option::take)
        .ok_or(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            reason: "floating tensor payload disagrees with target scalar type".to_string(),
        })
}

fn scalar_kind_name(item: RawJsonScalar) -> &'static str {
    match item {
        RawJsonScalar::Bool(_) => "boolean",
        RawJsonScalar::Integer(_) => "integer",
        RawJsonScalar::Float(_) => "float",
        RawJsonScalar::Unsupported(kind) => kind,
    }
}

pub(crate) fn decode_tensor_values<F: SKFloat>(
    tensor_name: &str,
    data_type: SKTensorDataType,
    raw_values: Vec<RawJsonScalar>,
) -> Result<SKTensorValues<F>, SKError> {
    match data_type {
        SKTensorDataType::F32 => {
            let mut parsed = Vec::with_capacity(raw_values.len());
            for (position, item) in raw_values.into_iter().enumerate() {
                let scalar = match item {
                    RawJsonScalar::Float(value) => value as f32,
                    RawJsonScalar::Integer(value) => value as f32,
                    other => {
                        return Err(SKError::InvalidHyperparameter {
                            name: "tensor_data_type",
                            reason: format!(
                                "tensor `{tensor_name}` at index {position} found `{}` instead of F32",
                                scalar_kind_name(other)
                            ),
                        });
                    }
                };
                if !scalar.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
                parsed.push(scalar);
            }
            Ok(SKTensorValues::Floating(cast_floating_vector(parsed)?))
        }
        SKTensorDataType::F64 => {
            let mut parsed = Vec::with_capacity(raw_values.len());
            for (position, item) in raw_values.into_iter().enumerate() {
                let scalar = match item {
                    RawJsonScalar::Float(value) => value,
                    RawJsonScalar::Integer(value) => value as f64,
                    other => {
                        return Err(SKError::InvalidHyperparameter {
                            name: "tensor_data_type",
                            reason: format!(
                                "tensor `{tensor_name}` at index {position} found `{}` instead of F64",
                                scalar_kind_name(other)
                            ),
                        });
                    }
                };
                if !scalar.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
                parsed.push(scalar);
            }
            Ok(SKTensorValues::Floating(cast_floating_vector(parsed)?))
        }
        SKTensorDataType::I64 => {
            let mut parsed = Vec::with_capacity(raw_values.len());
            for (position, item) in raw_values.into_iter().enumerate() {
                match item {
                    RawJsonScalar::Integer(value) => parsed.push(value),
                    other => {
                        return Err(SKError::InvalidHyperparameter {
                            name: "tensor_data_type",
                            reason: format!(
                                "tensor `{tensor_name}` at index {position} found `{}` instead of I64",
                                scalar_kind_name(other)
                            ),
                        });
                    }
                }
            }
            Ok(SKTensorValues::Integer(parsed))
        }
        SKTensorDataType::Bool => {
            let mut parsed = Vec::with_capacity(raw_values.len());
            for (position, item) in raw_values.into_iter().enumerate() {
                match item {
                    RawJsonScalar::Bool(flag) => parsed.push(flag),
                    other => {
                        return Err(SKError::InvalidHyperparameter {
                            name: "tensor_data_type",
                            reason: format!(
                                "tensor `{tensor_name}` at index {position} found `{}` instead of Bool",
                                scalar_kind_name(other)
                            ),
                        });
                    }
                }
            }
            Ok(SKTensorValues::Boolean(parsed))
        }
        _ => Err(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            reason: format!("unsupported tensor data type `{}`", data_type.as_str()),
        }),
    }
}
