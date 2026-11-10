//! The named tensor and its dtype (spec `model-export-contracts`).

use crate::SKError;
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
    fn dtype(&self) -> SKTensorDType {
        match self {
            SKTensorValues::Floating(_) => tensor_dtype_for_scalar::<F>(),
            SKTensorValues::Integer(_) => SKTensorDType::I64,
            SKTensorValues::Boolean(_) => SKTensorDType::Bool,
        }
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
