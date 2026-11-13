//! Safe little-endian tensor byte encoding and decoding without `unsafe`
//! (spec `safetensors-format`).

use std::any::Any;

use safetensors::Dtype;
use sciencekit_common::{SKError, SKFloat, SKTensor, SKTensorDataType, SKTensorValues};

use super::header::SK_SAFETENSORS_PADDING_TENSOR_NAME;

/// Map [`SKTensorDataType`] to the standard Safetensors [`Dtype`].
pub(crate) fn to_safetensors_dtype(data_type: SKTensorDataType) -> Result<Dtype, SKError> {
    match data_type {
        SKTensorDataType::F32 => Ok(Dtype::F32),
        SKTensorDataType::F64 => Ok(Dtype::F64),
        SKTensorDataType::I64 => Ok(Dtype::I64),
        SKTensorDataType::Bool => Ok(Dtype::BOOL),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unsupported SKTensorDataType variant",
            suggestion: "export F32, F64, I64 or Bool tensors",
        }),
    }
}

/// Map a Safetensors [`Dtype`] back to [`SKTensorDataType`].
pub(crate) fn from_safetensors_dtype(dtype: Dtype) -> Result<SKTensorDataType, SKError> {
    match dtype {
        Dtype::F32 => Ok(SKTensorDataType::F32),
        Dtype::F64 => Ok(SKTensorDataType::F64),
        Dtype::I64 => Ok(SKTensorDataType::I64),
        Dtype::BOOL => Ok(SKTensorDataType::Bool),
        other => Err(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            reason: format!(
                "unsupported Safetensors dtype `{other:?}`; expected F32, F64, I64 or BOOL"
            ),
        }),
    }
}

fn ensure_finite_slice<F: SKFloat>(elements: &[F]) -> Result<(), SKError> {
    elements
        .iter()
        .position(|value| !value.is_finite())
        .map_or(Ok(()), |position| Err(SKError::NonFiniteValue { position }))
}

/// Encode a single [`SKTensor`] into `(Dtype, shape, little_endian_bytes)`.
pub(crate) fn encode_tensor_payload<F: SKFloat>(
    tensor: &SKTensor<F>,
) -> Result<(Dtype, Vec<usize>, Vec<u8>), SKError> {
    if tensor.name() == SK_SAFETENSORS_PADDING_TENSOR_NAME {
        return Err(SKError::InvalidHyperparameter {
            name: "tensor_name",
            reason: format!(
                "tensor name `{SK_SAFETENSORS_PADDING_TENSOR_NAME}` is reserved for header padding"
            ),
        });
    }
    let dtype = to_safetensors_dtype(tensor.data_type())?;
    let bytes = match tensor.values() {
        SKTensorValues::Floating(elements) => {
            ensure_finite_slice(elements)?;
            let any_elements = elements as &dyn Any;
            if let Some(single) = any_elements.downcast_ref::<Vec<f32>>() {
                let mut buffer = Vec::with_capacity(single.len() * 4);
                single
                    .iter()
                    .for_each(|value| buffer.extend_from_slice(&value.to_le_bytes()));
                buffer
            } else if let Some(double) = any_elements.downcast_ref::<Vec<f64>>() {
                let mut buffer = Vec::with_capacity(double.len() * 8);
                double
                    .iter()
                    .for_each(|value| buffer.extend_from_slice(&value.to_le_bytes()));
                buffer
            } else {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown floating scalar in SKTensorValues",
                    suggestion: "export an f32 or f64 model state",
                });
            }
        }
        SKTensorValues::Integer(elements) => {
            let mut buffer = Vec::with_capacity(elements.len() * 8);
            elements
                .iter()
                .for_each(|value| buffer.extend_from_slice(&value.to_le_bytes()));
            buffer
        }
        SKTensorValues::Boolean(elements) => elements.iter().map(|&flag| u8::from(flag)).collect(),
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown SKTensorValues variant",
                suggestion: "export F32, F64, I64 or Bool tensors",
            });
        }
    };
    Ok((dtype, tensor.shape().to_vec(), bytes))
}

fn cast_floating_vector<Source: 'static, F: SKFloat>(
    values: Vec<Source>,
) -> Result<Vec<F>, SKError> {
    let mut slot = Some(values);
    let any_slot = &mut slot as &mut dyn Any;
    any_slot
        .downcast_mut::<Option<Vec<F>>>()
        .and_then(Option::take)
        .ok_or_else(|| SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            reason: "floating tensor dtype disagrees with target model scalar".to_string(),
        })
}

/// Decode raw little-endian Safetensors bytes into an [`SKTensor`].
pub(crate) fn decode_tensor_payload<F: SKFloat>(
    name: &str,
    dtype: Dtype,
    shape: &[usize],
    raw_bytes: &[u8],
    expected_scalar: SKTensorDataType,
) -> Result<SKTensor<F>, SKError> {
    let data_type = from_safetensors_dtype(dtype)?;
    if data_type.is_floating() && data_type != expected_scalar {
        return Err(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            reason: format!(
                "tensor `{name}` uses {} but the model scalar is {}",
                data_type.as_str(),
                expected_scalar.as_str()
            ),
        });
    }
    let values = match data_type {
        SKTensorDataType::F32 => {
            let mut decoded = Vec::with_capacity(raw_bytes.len() / 4);
            for (position, chunk) in raw_bytes.chunks_exact(4).enumerate() {
                let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                    SKError::Conversion("misaligned F32 chunk in Safetensors payload".to_string())
                })?;
                let scalar = f32::from_le_bytes(bytes);
                if !scalar.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
                decoded.push(scalar);
            }
            SKTensorValues::Floating(cast_floating_vector(decoded)?)
        }
        SKTensorDataType::F64 => {
            let mut decoded = Vec::with_capacity(raw_bytes.len() / 8);
            for (position, chunk) in raw_bytes.chunks_exact(8).enumerate() {
                let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                    SKError::Conversion("misaligned F64 chunk in Safetensors payload".to_string())
                })?;
                let scalar = f64::from_le_bytes(bytes);
                if !scalar.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
                decoded.push(scalar);
            }
            SKTensorValues::Floating(cast_floating_vector(decoded)?)
        }
        SKTensorDataType::I64 => {
            let decoded: Vec<i64> = raw_bytes
                .chunks_exact(8)
                .map(|chunk| {
                    let bytes: [u8; 8] = chunk.try_into().expect("chunks_exact guarantees 8 bytes");
                    i64::from_le_bytes(bytes)
                })
                .collect();
            SKTensorValues::Integer(decoded)
        }
        SKTensorDataType::Bool => {
            let mut decoded = Vec::with_capacity(raw_bytes.len());
            for (position, &byte) in raw_bytes.iter().enumerate() {
                match byte {
                    0 => decoded.push(false),
                    1 => decoded.push(true),
                    invalid => {
                        return Err(SKError::InvalidHyperparameter {
                            name: "tensor_data_type",
                            reason: format!(
                                "tensor `{name}` contains invalid boolean byte `{invalid}` at index {position}"
                            ),
                        });
                    }
                }
            }
            SKTensorValues::Boolean(decoded)
        }
        _ => {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_data_type",
                reason: format!("unsupported tensor data type `{}`", data_type.as_str()),
            });
        }
    };
    SKTensor::new(name, data_type, shape.to_vec(), values)
}
