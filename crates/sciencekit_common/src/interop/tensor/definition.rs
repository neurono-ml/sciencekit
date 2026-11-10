//! The named tensor definition.

use crate::SKError;
use crate::sk_float::SKFloat;

use super::dtype::{SKTensorDType, tensor_dtype_for_scalar};
use super::values::SKTensorValues;

/// A named, typed tensor with a shape and a value payload.
///
/// The name is a stable dotted path (`linear.coefficients`, `tree.nodes.feature`)
/// documented per model; format adapters preserve it verbatim. Values are owned
/// and immutable after construction.
#[derive(Debug, Clone, PartialEq)]
pub struct SKTensor<F: SKFloat> {
    name: String,
    dtype: SKTensorDType,
    shape: Vec<usize>,
    values: SKTensorValues<F>,
}

impl<F: SKFloat> SKTensor<F> {
    /// Build a tensor from an explicit dtype and payload, validating the name,
    /// the shape/value agreement and the dtype/payload agreement.
    pub fn new(
        name: impl Into<String>,
        dtype: SKTensorDType,
        shape: Vec<usize>,
        values: SKTensorValues<F>,
    ) -> Result<Self, SKError> {
        let name = name.into();
        if name.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_name",
                reason: "a tensor name must not be empty".to_string(),
            });
        }
        let expected_dtype = values.dtype();
        if dtype != expected_dtype {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_dtype",
                reason: format!(
                    "dtype {} does not match the `{}` payload",
                    dtype.as_str(),
                    expected_dtype.as_str()
                ),
            });
        }
        let declared = shape.iter().try_fold(1_usize, |product, dimension| {
            product.checked_mul(*dimension)
        });
        if declared != Some(values.len()) {
            return Err(SKError::ShapeMismatch {
                expected: shape,
                found: vec![values.len()],
            });
        }
        Ok(SKTensor {
            name,
            dtype,
            shape,
            values,
        })
    }

    /// Build a floating tensor in the model scalar.
    pub fn floating(
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<F>,
    ) -> Result<Self, SKError> {
        SKTensor::new(
            name,
            tensor_dtype_for_scalar::<F>(),
            shape,
            SKTensorValues::Floating(values),
        )
    }

    /// Build a signed 64-bit integer tensor.
    pub fn integer(
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<i64>,
    ) -> Result<Self, SKError> {
        SKTensor::new(
            name,
            SKTensorDType::I64,
            shape,
            SKTensorValues::Integer(values),
        )
    }

    /// Build a boolean tensor.
    pub fn boolean(
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<bool>,
    ) -> Result<Self, SKError> {
        SKTensor::new(
            name,
            SKTensorDType::Bool,
            shape,
            SKTensorValues::Boolean(values),
        )
    }

    /// The stable tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The element dtype.
    pub fn dtype(&self) -> SKTensorDType {
        self.dtype
    }

    /// The shape whose product equals [`Self::number_of_elements`].
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The number of stored values.
    pub fn number_of_elements(&self) -> usize {
        self.values.len()
    }

    /// The owned value payload.
    pub fn values(&self) -> &SKTensorValues<F> {
        &self.values
    }
}
