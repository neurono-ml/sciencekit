//! Tensor dtypes and the scalar-to-dtype mapping.

use crate::sk_float::SKFloat;

/// The element type stored in an exported tensor.
///
/// Floating tensors carry the model scalar; integer and boolean tensors carry
/// class tables and tree indices without forcing them into the scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SKTensorDType {
    /// Single-precision floating point.
    F32,
    /// Double-precision floating point.
    F64,
    /// Signed 64-bit integer.
    I64,
    /// Boolean.
    Bool,
}

impl SKTensorDType {
    /// The canonical textual name used by every format adapter.
    pub fn as_str(self) -> &'static str {
        match self {
            SKTensorDType::F32 => "F32",
            SKTensorDType::F64 => "F64",
            SKTensorDType::I64 => "I64",
            SKTensorDType::Bool => "Bool",
        }
    }

    /// Whether the dtype stores floating-point values.
    pub fn is_floating(self) -> bool {
        matches!(self, SKTensorDType::F32 | SKTensorDType::F64)
    }
}

/// The tensor dtype matching the model scalar `F`.
pub(crate) fn tensor_dtype_for_scalar<F: SKFloat>() -> SKTensorDType {
    if std::mem::size_of::<F>() == std::mem::size_of::<f32>() {
        SKTensorDType::F32
    } else {
        SKTensorDType::F64
    }
}
