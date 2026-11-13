//! Shared typed rejections for initializer shapes and dtypes
//! (spec `native-model-import`, requirement "Validation and concurrency").

use sciencekit_common::SKTensorDataType;
use sciencekit_interop::native::SKNativeImportError;

use crate::graph::SKOnnxTensorDType;
use crate::native::error::SKOnnxNativeError;

use super::initializers::SKOnnxConstant;

/// A typed shape rejection raised while mapping an ONNX graph.
pub(crate) fn sk_shape_mismatch(
    tensor: &str,
    expected: Vec<usize>,
    found: Vec<usize>,
) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(SKNativeImportError::ShapeMismatch {
        tensor: tensor.to_string(),
        expected,
        found,
    })
}

/// A typed dtype rejection raised while mapping an ONNX graph.
pub(crate) fn sk_dtype_mismatch(
    tensor: &str,
    expected: SKOnnxTensorDType,
    found: SKOnnxTensorDType,
) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(SKNativeImportError::DtypeMismatch {
        tensor: tensor.to_string(),
        expected: sk_tensor_data_type(expected),
        found: sk_tensor_data_type(found),
    })
}

/// Require a learned parameter to carry a floating-point element type.
pub(crate) fn sk_require_floating(
    constant: &SKOnnxConstant,
    name: &str,
) -> Result<(), SKOnnxNativeError> {
    if constant.is_floating() {
        return Ok(());
    }
    Err(sk_dtype_mismatch(
        name,
        SKOnnxTensorDType::Double,
        constant.data_type.unwrap_or(SKOnnxTensorDType::Int64),
    ))
}

/// The neutral data type matching a recognized ONNX element type.
fn sk_tensor_data_type(data_type: SKOnnxTensorDType) -> SKTensorDataType {
    match data_type {
        SKOnnxTensorDType::Float => SKTensorDataType::F32,
        SKOnnxTensorDType::Double => SKTensorDataType::F64,
        SKOnnxTensorDType::Int64 => SKTensorDataType::I64,
        SKOnnxTensorDType::Boolean => SKTensorDataType::Bool,
    }
}
