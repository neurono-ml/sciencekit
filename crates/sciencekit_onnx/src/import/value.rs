//! The dense tensor values the interpreter moves between operators.

use ndarray::{ArrayD, IxDyn};
use serde_onnx::ir::ElemType;

use super::SKOnnxImportError;

/// One intermediate ONNX tensor.
///
/// Only the four dense element types the documented operator subset needs are
/// represented; anything else is rejected while loading the model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SKOnnxValue {
    /// Single-precision floats.
    Float32(ArrayD<f32>),
    /// Double-precision floats.
    Float64(ArrayD<f64>),
    /// 64-bit integer indices and labels.
    Int64(ArrayD<i64>),
    /// Boolean comparison results.
    Boolean(ArrayD<bool>),
}

impl SKOnnxValue {
    /// The shared IR element type matching this value.
    pub(crate) fn element_type(&self) -> ElemType {
        match self {
            Self::Float32(_) => ElemType::Float,
            Self::Float64(_) => ElemType::Double,
            Self::Int64(_) => ElemType::Int64,
            Self::Boolean(_) => ElemType::Bool,
        }
    }

    /// The tensor shape.
    pub(crate) fn shape(&self) -> Vec<usize> {
        match self {
            Self::Float32(array) => array.shape().to_vec(),
            Self::Float64(array) => array.shape().to_vec(),
            Self::Int64(array) => array.shape().to_vec(),
            Self::Boolean(array) => array.shape().to_vec(),
        }
    }

    /// The number of stored elements.
    pub(crate) fn length(&self) -> usize {
        match self {
            Self::Float32(array) => array.len(),
            Self::Float64(array) => array.len(),
            Self::Int64(array) => array.len(),
            Self::Boolean(array) => array.len(),
        }
    }

    /// The values as `f64`, for the predictor boundary and float operators.
    pub(crate) fn to_f64_values(&self) -> Vec<f64> {
        match self {
            Self::Float32(array) => array.iter().map(|value| f64::from(*value)).collect(),
            Self::Float64(array) => array.iter().copied().collect(),
            Self::Int64(array) => array.iter().map(|value| *value as f64).collect(),
            Self::Boolean(array) => array
                .iter()
                .map(|value| if *value { 1.0 } else { 0.0 })
                .collect(),
        }
    }

    /// The values as `i64`, for label outputs and index operands.
    pub(crate) fn to_i64_values(&self) -> Vec<i64> {
        match self {
            Self::Int64(array) => array.iter().copied().collect(),
            Self::Boolean(array) => array.iter().map(|value| i64::from(*value)).collect(),
            Self::Float32(array) => array.iter().map(|value| *value as i64).collect(),
            Self::Float64(array) => array.iter().map(|value| *value as i64).collect(),
        }
    }

    /// Build a value from `f64` storage in the shared element type.
    pub(crate) fn from_f64_values(
        element_type: ElemType,
        shape: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, SKOnnxImportError> {
        let dimension = IxDyn(&shape);
        match element_type {
            ElemType::Float | ElemType::Float16 | ElemType::Bfloat16 => {
                Ok(Self::Float32(shape_vec(
                    dimension,
                    values.into_iter().map(|value| value as f32).collect(),
                )?))
            }
            ElemType::Double => Ok(Self::Float64(shape_vec(dimension, values)?)),
            ElemType::Int64 => Ok(Self::Int64(shape_vec(
                dimension,
                values.into_iter().map(|value| value as i64).collect(),
            )?)),
            ElemType::Bool => Ok(Self::Boolean(shape_vec(
                dimension,
                values.into_iter().map(|value| value != 0.0).collect(),
            )?)),
            other => Err(SKOnnxImportError::UnsupportedDtype {
                element_type: other.name().to_string(),
            }),
        }
    }

    /// Convert this value to another dense element type.
    pub(crate) fn cast_to(&self, element_type: ElemType) -> Result<Self, SKOnnxImportError> {
        let shape = self.shape();
        let dimension = IxDyn(&shape);
        match element_type {
            ElemType::Float | ElemType::Float16 | ElemType::Bfloat16 => {
                Ok(Self::Float32(shape_vec(
                    dimension,
                    self.to_f64_values()
                        .into_iter()
                        .map(|value| value as f32)
                        .collect(),
                )?))
            }
            ElemType::Double => Ok(Self::Float64(shape_vec(dimension, self.to_f64_values())?)),
            ElemType::Int64 => Ok(Self::Int64(shape_vec(dimension, self.to_i64_values())?)),
            ElemType::Bool => Ok(Self::Boolean(shape_vec(
                dimension,
                self.to_f64_values()
                    .into_iter()
                    .map(|value| value != 0.0)
                    .collect(),
            )?)),
            other => Err(SKOnnxImportError::UnsupportedDtype {
                element_type: other.name().to_string(),
            }),
        }
    }
}

fn shape_vec<T>(dimension: IxDyn, values: Vec<T>) -> Result<ArrayD<T>, SKOnnxImportError> {
    ArrayD::from_shape_vec(dimension, values).map_err(|error| SKOnnxImportError::InvalidModel {
        detail: error.to_string(),
    })
}
