//! Library-owned output buffers released by `sk_ffi_array_free`.

use ndarray::{Array1, Array2};

use super::definition::SKFfiDataType;

/// An owned output payload matching the descriptor dtype.
#[derive(Debug, Clone, PartialEq)]
pub enum SKFfiOwnedValues {
    /// Single-precision floating values.
    F32(Vec<f32>),
    /// Double-precision floating values.
    F64(Vec<f64>),
    /// Signed 64-bit integer values.
    I64(Vec<i64>),
    /// Unsigned byte values.
    U8(Vec<u8>),
}

impl SKFfiOwnedValues {
    /// The number of stored values.
    pub fn len(&self) -> usize {
        match self {
            SKFfiOwnedValues::F32(values) => values.len(),
            SKFfiOwnedValues::F64(values) => values.len(),
            SKFfiOwnedValues::I64(values) => values.len(),
            SKFfiOwnedValues::U8(values) => values.len(),
        }
    }

    /// Whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An output array the library allocated and the caller releases with
/// `sk_ffi_array_free`.
#[derive(Debug, Clone, PartialEq)]
pub struct SKFfiOwnedArray {
    dtype: SKFfiDataType,
    shape: Vec<usize>,
    values: SKFfiOwnedValues,
}

impl SKFfiOwnedArray {
    /// Build an owned f64 vector with shape `[len]`.
    pub fn from_f64_vector(values: Array1<f64>) -> SKFfiOwnedArray {
        let shape = vec![values.len()];
        SKFfiOwnedArray {
            dtype: SKFfiDataType::F64,
            shape,
            values: SKFfiOwnedValues::F64(values.to_vec()),
        }
    }

    /// Build an owned f32 vector with shape `[len]`.
    pub fn from_f32_vector(values: Array1<f32>) -> SKFfiOwnedArray {
        let shape = vec![values.len()];
        SKFfiOwnedArray {
            dtype: SKFfiDataType::F32,
            shape,
            values: SKFfiOwnedValues::F32(values.to_vec()),
        }
    }

    /// Build an owned f64 matrix with shape `[rows, columns]`.
    pub fn from_f64_matrix(values: Array2<f64>) -> SKFfiOwnedArray {
        let shape = vec![values.nrows(), values.ncols()];
        SKFfiOwnedArray {
            dtype: SKFfiDataType::F64,
            shape,
            values: SKFfiOwnedValues::F64(values.iter().copied().collect()),
        }
    }

    /// Build an owned f32 matrix with shape `[rows, columns]`.
    pub fn from_f32_matrix(values: Array2<f32>) -> SKFfiOwnedArray {
        let shape = vec![values.nrows(), values.ncols()];
        SKFfiOwnedArray {
            dtype: SKFfiDataType::F32,
            shape,
            values: SKFfiOwnedValues::F32(values.iter().copied().collect()),
        }
    }

    /// The element data type.
    pub fn dtype(&self) -> SKFfiDataType {
        self.dtype
    }

    /// The shape in row-major order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The owned payload.
    pub fn values(&self) -> &SKFfiOwnedValues {
        &self.values
    }

    /// Consume the array, returning its parts for the boundary to publish.
    pub fn into_parts(self) -> (SKFfiDataType, Vec<usize>, SKFfiOwnedValues) {
        (self.dtype, self.shape, self.values)
    }
}
