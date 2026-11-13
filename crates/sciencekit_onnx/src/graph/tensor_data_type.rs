//! The ONNX tensor element types.
//!
//! The neutral model owns its own data-type vocabulary; this module is the
//! single place where it meets the shared [`serde_onnx::ir::ElemType`] used by
//! both the exporter and the importer.

use sciencekit_common::{SKError, SKTensorDataType};
use serde_onnx::ir::ElemType;

/// The ONNX `TensorProto.DataType` values the exporter emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKOnnxTensorDType {
    /// `TensorProto.DataType.FLOAT` (1), backed by little-endian `f32`.
    Float,
    /// `TensorProto.DataType.DOUBLE` (11), backed by little-endian `f64`.
    Double,
    /// `TensorProto.DataType.INT64` (7), backed by little-endian `i64`.
    Int64,
    /// `TensorProto.DataType.BOOL` (9), backed by one byte per value.
    Boolean,
}

impl SKOnnxTensorDType {
    /// The numeric `TensorProto.DataType` value.
    pub fn element_type(self) -> i32 {
        self.to_element_type().disc()
    }

    /// The shared IR element type.
    pub fn to_element_type(self) -> ElemType {
        match self {
            SKOnnxTensorDType::Float => ElemType::Float,
            SKOnnxTensorDType::Double => ElemType::Double,
            SKOnnxTensorDType::Int64 => ElemType::Int64,
            SKOnnxTensorDType::Boolean => ElemType::Bool,
        }
    }

    /// Recover the exporter vocabulary from a shared IR element type.
    pub fn from_element_type(element_type: ElemType) -> Result<Self, SKError> {
        match element_type {
            ElemType::Float => Ok(SKOnnxTensorDType::Float),
            ElemType::Double => Ok(SKOnnxTensorDType::Double),
            ElemType::Int64 => Ok(SKOnnxTensorDType::Int64),
            ElemType::Bool => Ok(SKOnnxTensorDType::Boolean),
            other => Err(SKError::UnsupportedOperation {
                operation: "onnx_import",
                reason: format!(
                    "the ONNX element type `{}` is outside the importer's dense numeric subset",
                    other.name()
                ),
            }),
        }
    }

    /// Whether the element type stores floating-point values.
    pub fn is_floating(self) -> bool {
        matches!(self, SKOnnxTensorDType::Float | SKOnnxTensorDType::Double)
    }

    /// The dtype matching a neutral model tensor's data type.
    pub fn from_tensor_data_type(data_type: SKTensorDataType) -> Result<Self, SKError> {
        match data_type {
            SKTensorDataType::F32 => Ok(SKOnnxTensorDType::Float),
            SKTensorDataType::F64 => Ok(SKOnnxTensorDType::Double),
            SKTensorDataType::I64 => Ok(SKOnnxTensorDType::Int64),
            SKTensorDataType::Bool => Ok(SKOnnxTensorDType::Boolean),
            other => Err(SKError::UnsupportedOperation {
                operation: "onnx_export",
                reason: format!(
                    "the ONNX exporter does not map the model tensor data type `{}`",
                    other.as_str()
                ),
            }),
        }
    }

    /// The dtype matching the model scalar `F`.
    pub fn for_scalar<F: sciencekit_common::SKFloat>() -> Self {
        if std::mem::size_of::<F>() == std::mem::size_of::<f32>() {
            SKOnnxTensorDType::Float
        } else {
            SKOnnxTensorDType::Double
        }
    }
}
