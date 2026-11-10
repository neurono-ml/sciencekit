//! The owned tensor payload matching a dtype.

use crate::sk_float::SKFloat;

use super::dtype::{SKTensorDType, tensor_dtype_for_scalar};

/// The owned payload of a tensor, matching its [`SKTensorDType`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SKTensorValues<F: SKFloat> {
    /// Floating-point values in the model scalar.
    Floating(Vec<F>),
    /// Signed 64-bit integer values.
    Integer(Vec<i64>),
    /// Boolean values.
    Boolean(Vec<bool>),
}

impl<F: SKFloat> SKTensorValues<F> {
    /// The number of stored values.
    pub fn len(&self) -> usize {
        match self {
            SKTensorValues::Floating(values) => values.len(),
            SKTensorValues::Integer(values) => values.len(),
            SKTensorValues::Boolean(values) => values.len(),
        }
    }

    /// Whether the payload holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The dtype implied by the payload variant.
    pub(crate) fn dtype(&self) -> SKTensorDType {
        match self {
            SKTensorValues::Floating(_) => tensor_dtype_for_scalar::<F>(),
            SKTensorValues::Integer(_) => SKTensorDType::I64,
            SKTensorValues::Boolean(_) => SKTensorDType::Bool,
        }
    }
}
