//! Length and stride validation against a shape.

use super::error::SKFfiArrayError;
use super::shape::SKFfiArrayShape;

/// Require the declared element count to equal the shape product.
pub fn sk_ffi_validate_length(
    shape: &SKFfiArrayShape,
    length: i64,
) -> Result<usize, SKFfiArrayError> {
    if length < 0 {
        return Err(SKFfiArrayError::NegativeLength(length));
    }
    let expected = shape.number_of_elements()?;
    if length as usize != expected {
        return Err(SKFfiArrayError::LengthMismatch {
            expected: expected as i64,
            found: length,
        });
    }
    Ok(expected)
}

/// Accept null strides (row-major) or exactly the contiguous element strides.
///
/// The last dimension strides by one element; every earlier dimension strides
/// by the product of the dimensions after it.
pub fn sk_ffi_validate_contiguous_strides(
    strides: Option<&[i64]>,
    shape: &SKFfiArrayShape,
) -> Result<(), SKFfiArrayError> {
    let Some(strides) = strides else {
        return Ok(());
    };
    let dimensions = shape.dimensions();
    if strides.len() != dimensions.len() {
        return Err(SKFfiArrayError::NonContiguousStrides);
    }
    let mut expected = 1_i64;
    for (dimension, &stride) in dimensions.iter().zip(strides.iter()).rev() {
        if stride != expected {
            return Err(SKFfiArrayError::NonContiguousStrides);
        }
        expected = expected
            .checked_mul(*dimension as i64)
            .ok_or(SKFfiArrayError::LengthOutOfRange)?;
    }
    Ok(())
}
