//! Publishing library-owned output buffers and releasing them again.

use std::ffi::c_void;
use std::ptr;
use std::slice;

use crate::array::{SKFfiArray, SKFfiDataType, SKFfiOwnedArray, SKFfiOwnedValues};
use crate::error_kind::SKFfiError;

/// Publish an owned output array as a C descriptor.
///
/// # Safety
///
/// The caller stores the returned descriptor in a location the caller keeps
/// alive and eventually passes to [`sk_ffi_release_output`].
pub(crate) unsafe fn sk_ffi_publish_output(
    array: SKFfiOwnedArray,
) -> Result<SKFfiArray, SKFfiError> {
    let (dtype, shape, values) = array.into_parts();
    let dimensions: Vec<i64> = shape.iter().map(|dimension| *dimension as i64).collect();
    let ndim = dimensions.len() as i32;
    let length = values.len();
    let shape_buffer = dimensions.into_boxed_slice();
    let shape_pointer = Box::into_raw(shape_buffer) as *const i64;
    let data_pointer = match values {
        SKFfiOwnedValues::F32(entries) => {
            Box::into_raw(entries.into_boxed_slice()) as *const c_void
        }
        SKFfiOwnedValues::F64(entries) => {
            Box::into_raw(entries.into_boxed_slice()) as *const c_void
        }
        SKFfiOwnedValues::I64(entries) => {
            Box::into_raw(entries.into_boxed_slice()) as *const c_void
        }
        SKFfiOwnedValues::U8(entries) => Box::into_raw(entries.into_boxed_slice()) as *const c_void,
    };
    Ok(SKFfiArray {
        dtype: dtype as i32,
        ndim,
        shape: shape_pointer,
        strides: ptr::null(),
        data: data_pointer,
        len: length as i64,
    })
}

/// Release a descriptor produced by [`sk_ffi_publish_output`].
///
/// # Safety
///
/// `array` must point to a writable descriptor whose buffers came from
/// [`sk_ffi_publish_output`] and were not released before.
pub(crate) unsafe fn sk_ffi_release_output(array: *mut SKFfiArray) -> Result<(), SKFfiError> {
    if array.is_null() {
        return Err(SKFfiError::invalid_argument(
            "output descriptor pointer is null",
        ));
    }
    // SAFETY: the caller guarantees a writable descriptor.
    let descriptor = unsafe { &mut *array };
    if descriptor.data.is_null() && descriptor.shape.is_null() {
        return Ok(());
    }
    if descriptor.len < 0 {
        return Err(SKFfiError::invalid_argument(
            "output descriptor len is negative",
        ));
    }
    let length = descriptor.len as usize;
    let dtype = if descriptor.data.is_null() {
        None
    } else {
        Some(SKFfiDataType::from_code(descriptor.dtype).ok_or_else(|| {
            SKFfiError::invalid_argument("output descriptor has an unknown dtype")
        })?)
    };
    if !descriptor.shape.is_null() {
        if descriptor.ndim < 1 {
            return Err(SKFfiError::invalid_argument(
                "output descriptor ndim is not positive",
            ));
        }
        let ndim = descriptor.ndim as usize;
        // SAFETY: the shape pointer came from `publish_output` with `ndim`.
        let shape = unsafe {
            Box::from_raw(slice::from_raw_parts_mut(
                descriptor.shape as *mut i64,
                ndim,
            ))
        };
        drop(shape);
        descriptor.shape = ptr::null();
    }
    if let Some(dtype) = dtype {
        // SAFETY: the data pointer came from `publish_output` with `length`.
        unsafe {
            match dtype {
                SKFfiDataType::F32 => {
                    drop(Box::from_raw(slice::from_raw_parts_mut(
                        descriptor.data as *mut f32,
                        length,
                    )));
                }
                SKFfiDataType::F64 => {
                    drop(Box::from_raw(slice::from_raw_parts_mut(
                        descriptor.data as *mut f64,
                        length,
                    )));
                }
                SKFfiDataType::I64 => {
                    drop(Box::from_raw(slice::from_raw_parts_mut(
                        descriptor.data as *mut i64,
                        length,
                    )));
                }
                SKFfiDataType::U8 => {
                    drop(Box::from_raw(slice::from_raw_parts_mut(
                        descriptor.data as *mut u8,
                        length,
                    )));
                }
            }
        }
        descriptor.data = ptr::null();
    }
    descriptor.len = 0;
    descriptor.ndim = 0;
    Ok(())
}
