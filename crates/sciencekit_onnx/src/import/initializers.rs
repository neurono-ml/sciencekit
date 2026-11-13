//! Converting declared initializers into interpreter values.

use ndarray::ArrayD;
use serde_onnx::ir::{Tensor, TensorData, TensorStorage};

use super::SKOnnxImportError;
use super::value::SKOnnxValue;

/// Convert one initializer into an interpreter value.
pub(crate) fn tensor_value(tensor: &Tensor) -> Result<SKOnnxValue, SKOnnxImportError> {
    if let TensorStorage::External(_) = tensor.data {
        return Err(SKOnnxImportError::ExternalData {
            tensor: tensor.name.clone(),
        });
    }
    let shape: Vec<usize> = tensor
        .shape
        .iter()
        .map(|dimension| dimension.fixed_value().unwrap_or(0).max(0) as usize)
        .collect();
    let name = tensor.name.as_str();
    match tensor.data() {
        Some(TensorData::F32(values)) => {
            Ok(SKOnnxValue::Float32(build(&shape, values.clone(), name)?))
        }
        Some(TensorData::F64(values)) => {
            Ok(SKOnnxValue::Float64(build(&shape, values.clone(), name)?))
        }
        Some(TensorData::I64(values)) => {
            Ok(SKOnnxValue::Int64(build(&shape, values.clone(), name)?))
        }
        Some(TensorData::I32(values)) => Ok(SKOnnxValue::Int64(build(
            &shape,
            values.iter().map(|value| i64::from(*value)).collect(),
            name,
        )?)),
        Some(TensorData::Bool(values)) => {
            Ok(SKOnnxValue::Boolean(build(&shape, values.clone(), name)?))
        }
        Some(TensorData::F16(bits)) => Ok(SKOnnxValue::Float32(build(
            &shape,
            bits.iter()
                .map(|bits| f32::from(half::f16::from_bits(*bits)))
                .collect(),
            name,
        )?)),
        Some(other) => Err(SKOnnxImportError::UnsupportedDtype {
            element_type: other.elem_type().name().to_string(),
        }),
        None => Err(SKOnnxImportError::ExternalData {
            tensor: tensor.name.clone(),
        }),
    }
}

fn build<T>(shape: &[usize], values: Vec<T>, name: &str) -> Result<ArrayD<T>, SKOnnxImportError> {
    ArrayD::from_shape_vec(ndarray::IxDyn(shape), values).map_err(|error| {
        SKOnnxImportError::InvalidModel {
            detail: format!("initializer `{name}`: {error}"),
        }
    })
}
