//! The ONNX tensor element types.

use sciencekit_common::{SKError, SKTensorDataType};

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
        match self {
            SKOnnxTensorDType::Float => 1,
            SKOnnxTensorDType::Double => 11,
            SKOnnxTensorDType::Int64 => 7,
            SKOnnxTensorDType::Boolean => 9,
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
