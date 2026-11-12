//! Validation of a borrowed descriptor and the typed views it yields.
//!
//! The pointer must be valid for the duration of the call; for inputs, the
//! backing buffer must stay alive and immutable until the call returns.

use std::slice;

use ndarray::{ArrayView1, ArrayView2};

use crate::array::{
    SKFfiArray, SKFfiArrayError, SKFfiArrayShape, SKFfiDataType,
    sk_ffi_validate_contiguous_strides, sk_ffi_validate_length,
};
use crate::error_kind::SKFfiError;

/// A descriptor validated and copied out of C memory.
#[derive(Debug)]
pub(crate) struct SKFfiValidatedDescriptor<'a> {
    dtype: SKFfiDataType,
    shape: SKFfiArrayShape,
    data: &'a [u8],
    length: usize,
}

impl<'a> SKFfiValidatedDescriptor<'a> {
    /// Borrow the descriptor as a row-major `f64` matrix.
    pub(crate) fn matrix_f64(&self) -> Result<ArrayView2<'a, f64>, SKFfiError> {
        let (rows, columns) = self.require_matrix(SKFfiDataType::F64)?;
        // SAFETY: the descriptor validated a contiguous buffer of exactly
        // `length` f64 elements, and `data` remains borrowed for `'a`.
        let values: &'a [f64] =
            unsafe { slice::from_raw_parts(self.data.as_ptr() as *const f64, self.length) };
        ArrayView2::from_shape((rows, columns), values).map_err(SKFfiError::invalid_argument)
    }

    /// Borrow the descriptor as a row-major `f32` matrix.
    pub(crate) fn matrix_f32(&self) -> Result<ArrayView2<'a, f32>, SKFfiError> {
        let (rows, columns) = self.require_matrix(SKFfiDataType::F32)?;
        // SAFETY: the descriptor validated a contiguous buffer of exactly
        // `length` f32 elements, and `data` remains borrowed for `'a`.
        let values: &'a [f32] =
            unsafe { slice::from_raw_parts(self.data.as_ptr() as *const f32, self.length) };
        ArrayView2::from_shape((rows, columns), values).map_err(SKFfiError::invalid_argument)
    }

    /// Borrow the descriptor as a contiguous `f64` vector.
    pub(crate) fn vector_f64(&self) -> Result<ArrayView1<'a, f64>, SKFfiError> {
        let length = self.require_vector(SKFfiDataType::F64)?;
        // SAFETY: the descriptor validated a contiguous buffer of exactly
        // `length` f64 elements, and `data` remains borrowed for `'a`.
        let values: &'a [f64] =
            unsafe { slice::from_raw_parts(self.data.as_ptr() as *const f64, length) };
        Ok(ArrayView1::from(values))
    }

    /// Borrow the descriptor as a contiguous `f32` vector.
    pub(crate) fn vector_f32(&self) -> Result<ArrayView1<'a, f32>, SKFfiError> {
        let length = self.require_vector(SKFfiDataType::F32)?;
        // SAFETY: the descriptor validated a contiguous buffer of exactly
        // `length` f32 elements, and `data` remains borrowed for `'a`.
        let values: &'a [f32] =
            unsafe { slice::from_raw_parts(self.data.as_ptr() as *const f32, length) };
        Ok(ArrayView1::from(values))
    }

    fn require_matrix(&self, dtype: SKFfiDataType) -> Result<(usize, usize), SKFfiError> {
        self.require_dtype(dtype)?;
        Ok(self.shape.matrix_shape()?)
    }

    fn require_vector(&self, dtype: SKFfiDataType) -> Result<usize, SKFfiError> {
        self.require_dtype(dtype)?;
        Ok(self.shape.vector_length()?)
    }

    fn require_dtype(&self, expected: SKFfiDataType) -> Result<(), SKFfiError> {
        if self.dtype == expected {
            return Ok(());
        }
        Err(SKFfiArrayError::DtypeMismatch {
            expected: crate::model::dtype_name(expected),
            found: crate::model::dtype_name(self.dtype),
        }
        .into())
    }
}

/// Validate a borrowed descriptor before any element is read.
///
/// # Safety
///
/// `array` must point to a readable `SKFfiArray`; `shape` and, when non-null,
/// `strides` must point to `ndim` readable `i64` values; `data` must point to
/// `len` elements of the declared dtype that stay alive and immutable for
/// `'a`.
pub(crate) unsafe fn sk_ffi_validate_descriptor<'a>(
    array: *const SKFfiArray,
) -> Result<SKFfiValidatedDescriptor<'a>, SKFfiError> {
    if array.is_null() {
        return Err(SKFfiError::invalid_argument(
            "array descriptor pointer is null",
        ));
    }
    // SAFETY: the caller guarantees a readable descriptor.
    let descriptor = unsafe { *array };
    if descriptor.shape.is_null() {
        return Err(SKFfiArrayError::NullShape.into());
    }
    if descriptor.data.is_null() {
        return Err(SKFfiArrayError::NullData.into());
    }
    let shape_alignment = std::mem::align_of::<i64>();
    if (descriptor.shape as usize) % shape_alignment != 0 {
        return Err(SKFfiArrayError::MisalignedPointer { field: "shape" }.into());
    }
    if !(1..=2).contains(&descriptor.ndim) {
        return Err(SKFfiArrayError::InvalidNumberOfDimensions(descriptor.ndim).into());
    }
    // SAFETY: the caller guarantees `ndim` readable dimensions.
    let dimensions = unsafe { slice::from_raw_parts(descriptor.shape, descriptor.ndim as usize) };
    let shape = SKFfiArrayShape::new(dimensions)?;
    let length = sk_ffi_validate_length(&shape, descriptor.len)?;
    if !descriptor.strides.is_null() && (descriptor.strides as usize) % shape_alignment != 0 {
        return Err(SKFfiArrayError::MisalignedPointer { field: "strides" }.into());
    }
    let strides = if descriptor.strides.is_null() {
        None
    } else {
        // SAFETY: the pointer is non-null, checked for alignment, and the caller
        // guarantees `ndim` readable strides.
        Some(unsafe { slice::from_raw_parts(descriptor.strides, descriptor.ndim as usize) })
    };
    sk_ffi_validate_contiguous_strides(strides, &shape)?;
    let dtype = descriptor.data_type()?;
    if (descriptor.data as usize) % dtype.element_size() != 0 {
        return Err(SKFfiArrayError::MisalignedPointer { field: "data" }.into());
    }
    let byte_length = length
        .checked_mul(dtype.element_size())
        .ok_or(SKFfiArrayError::LengthOutOfRange)?;
    // SAFETY: the caller guarantees `length` elements of `dtype` at `data`.
    let data = unsafe { slice::from_raw_parts(descriptor.data as *const u8, byte_length) };
    Ok(SKFfiValidatedDescriptor {
        dtype,
        shape,
        data,
        length,
    })
}
