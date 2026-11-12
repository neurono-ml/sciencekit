//! The owned tensor payload matching a data_type.

use crate::sk_float::SKFloat;

use super::data_type::{SKTensorDataType, tensor_data_type_for_scalar};

/// The owned payload of a tensor, matching its [`SKTensorDataType`].
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

    /// The data_type implied by the payload variant.
    pub(crate) fn data_type(&self) -> SKTensorDataType {
        match self {
            SKTensorValues::Floating(_) => tensor_data_type_for_scalar::<F>(),
            SKTensorValues::Integer(_) => SKTensorDataType::I64,
            SKTensorValues::Boolean(_) => SKTensorDataType::Bool,
        }
    }
}
