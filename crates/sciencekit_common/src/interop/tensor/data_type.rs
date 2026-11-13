//! Tensor dtypes and the scalar-to-data_type mapping.

use crate::sk_float::SKFloat;

/// The element type stored in an exported tensor.
///
/// Floating tensors carry the model scalar; integer and boolean tensors carry
/// class tables and tree indices without forcing them into the scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SKTensorDataType {
    /// Single-precision floating point.
    F32,
    /// Double-precision floating point.
    F64,
    /// Signed 64-bit integer.
    I64,
    /// Boolean.
    Bool,
}

impl SKTensorDataType {
    /// The canonical textual name used by every format adapter.
    pub fn as_str(self) -> &'static str {
        match self {
            SKTensorDataType::F32 => "F32",
            SKTensorDataType::F64 => "F64",
            SKTensorDataType::I64 => "I64",
            SKTensorDataType::Bool => "Bool",
        }
    }

    /// Whether the data_type stores floating-point values.
    pub fn is_floating(self) -> bool {
        matches!(self, SKTensorDataType::F32 | SKTensorDataType::F64)
    }
}

/// The tensor data_type matching the model scalar `F`.
pub(crate) fn tensor_data_type_for_scalar<F: SKFloat>() -> SKTensorDataType {
    if std::mem::size_of::<F>() == std::mem::size_of::<f32>() {
        SKTensorDataType::F32
    } else {
        SKTensorDataType::F64
    }
}
