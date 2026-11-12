//! The borrowed array descriptor shared with C (spec `ffi-abi`).

use std::ffi::c_void;

use super::validation::{SKFfiArrayError, SKFfiArrayShape, sk_ffi_validate_contiguous_strides};

/// Element data types a descriptor may carry.
///
/// Floating descriptors drive modeling; the integer and byte variants exist so
/// future label and categorical payloads use one convention.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKFfiDataType {
    /// Single-precision floating point.
    F32 = 0,
    /// Double-precision floating point.
    F64 = 1,
    /// Signed 64-bit integer.
    I64 = 2,
    /// Unsigned byte.
    U8 = 3,
}

impl SKFfiDataType {
    /// Resolve the code a binding sent, rejecting unknown values.
    pub fn from_code(code: i32) -> Option<SKFfiDataType> {
        match code {
            0 => Some(SKFfiDataType::F32),
            1 => Some(SKFfiDataType::F64),
            2 => Some(SKFfiDataType::I64),
            3 => Some(SKFfiDataType::U8),
            _ => None,
        }
    }

    /// The byte width of one element.
    pub fn element_size(self) -> usize {
        match self {
            SKFfiDataType::F32 => 4,
            SKFfiDataType::F64 => 8,
            SKFfiDataType::I64 => 8,
            SKFfiDataType::U8 => 1,
        }
    }

    /// Whether the descriptor carries floating-point values.
    pub fn is_floating(self) -> bool {
        matches!(self, SKFfiDataType::F32 | SKFfiDataType::F64)
    }
}

/// A borrowed, C-layout array descriptor.
///
/// # Fields
///
/// - `dtype`: one of the [`SKFfiDataType`] codes.
/// - `ndim`: the number of dimensions (1 or 2 for every operation).
/// - `shape`: `ndim` non-negative dimensions, never null.
/// - `strides`: element strides, or null for a C-contiguous (row-major) array.
///   Non-null strides must describe the same row-major layout; other layouts
///   are rejected with [`SKFfiStatus::Unsupported`](crate::SKFfiStatus).
/// - `data`: the first element, never null.
/// - `len`: the total number of elements; it must equal the product of
///   `shape`.
///
/// # Ownership
///
/// The library reads `data` without copying for the duration of one call. The
/// caller keeps ownership and may free the buffer when the call returns.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SKFfiArray {
    /// Element data type.
    pub dtype: i32,
    /// Number of dimensions.
    pub ndim: i32,
    /// Pointer to `ndim` dimensions.
    pub shape: *const i64,
    /// Element strides, or null for C-contiguous layout.
    pub strides: *const i64,
    /// Pointer to the first element.
    pub data: *const c_void,
    /// Total element count.
    pub len: i64,
}

impl SKFfiArray {
    /// The data type, or an invalid-dtype error.
    pub fn data_type(&self) -> Result<SKFfiDataType, SKFfiArrayError> {
        SKFfiDataType::from_code(self.dtype).ok_or(SKFfiArrayError::UnknownDataType(self.dtype))
    }

    /// Validate a shape slice read from C (safe: the caller already copied it).
    pub fn validate_shape(&self, dimensions: &[i64]) -> Result<SKFfiArrayShape, SKFfiArrayError> {
        SKFfiArrayShape::new(dimensions)
    }

    /// Validate element strides (in elements) against a shape.
    pub fn validate_strides(
        &self,
        strides: Option<&[i64]>,
        shape: &SKFfiArrayShape,
    ) -> Result<(), SKFfiArrayError> {
        sk_ffi_validate_contiguous_strides(strides, shape)
    }
}
