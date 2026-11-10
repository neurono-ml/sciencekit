//! Reading published ABI output descriptors into owned matrices.

use std::ptr;
use std::slice;

use sciencekit_ffi::{SKFfiArray, SKFfiDataType, SKFfiStatus};

use crate::array::SKLuaMatrix;
use crate::error::SKLuaError;

/// A zero-initialised output slot the ABI may publish into.
pub(crate) fn sk_lua_empty_output() -> SKFfiArray {
    SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 0,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    }
}

/// Copy a published output into a matrix and release the ABI buffers.
pub(crate) fn sk_lua_take_output(
    status: SKFfiStatus,
    mut output: SKFfiArray,
) -> Result<SKLuaMatrix, SKLuaError> {
    if status != SKFfiStatus::Ok {
        // The ABI publishes nothing on failure; releasing the zero slot is a
        // documented no-op.
        let _ = sciencekit_ffi::sk_ffi_array_free(&mut output);
        return Err(SKLuaError::from_last_error(status));
    }
    let copied = sk_lua_copy_output(&output);
    let released = sciencekit_ffi::sk_ffi_array_free(&mut output);
    let matrix = copied?;
    if released != SKFfiStatus::Ok {
        return Err(SKLuaError::from_last_error(released));
    }
    Ok(matrix)
}

/// Read a published output descriptor into an owned matrix.
///
/// Predictions publish a one-dimensional vector; transforms publish a
/// two-dimensional matrix. Both become a matrix so the Lua layer has a single
/// output shape (a table of rows).
fn sk_lua_copy_output(output: &SKFfiArray) -> Result<SKLuaMatrix, SKLuaError> {
    if output.data.is_null() || output.shape.is_null() {
        return Err(SKLuaError::new(
            SKFfiStatus::Internal,
            "the ABI published an empty output descriptor",
        ));
    }
    if SKFfiDataType::from_code(output.dtype) != Some(SKFfiDataType::F64) {
        return Err(SKLuaError::unsupported(
            "the Lua binding only models double-precision outputs",
        ));
    }
    // SAFETY: the ABI published `ndim` dimensions and `len` f64 values for
    // this descriptor, and the buffers stay alive until `sk_ffi_array_free`.
    let dimensions = unsafe { slice::from_raw_parts(output.shape, output.ndim as usize) };
    let values = unsafe { slice::from_raw_parts(output.data as *const f64, output.len as usize) };
    match dimensions {
        [length] => Ok(SKLuaMatrix::new(*length as usize, 1, values.to_vec())),
        [rows, columns] => Ok(SKLuaMatrix::new(
            *rows as usize,
            *columns as usize,
            values.to_vec(),
        )),
        _ => Err(SKLuaError::new(
            SKFfiStatus::Internal,
            "the ABI published an output with an unexpected rank",
        )),
    }
}
