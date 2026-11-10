//! Tests for pointer validation and owned-buffer release (spec `ffi-abi`).

use std::ptr;

use super::{sk_ffi_publish_output, sk_ffi_release_output, sk_ffi_validate_descriptor};
use crate::array::{SKFfiArray, SKFfiArrayError, SKFfiDataType, SKFfiOwnedArray};

/// A contiguous f64 descriptor is read without copying.
#[test]
fn boundary_borrows_a_contiguous_f64_matrix() {
    let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2_i64, 3];
    let descriptor = SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 2,
        shape: shape.as_ptr(),
        strides: ptr::null(),
        data: values.as_ptr() as *const std::ffi::c_void,
        len: 6,
    };
    // SAFETY: every pointer points at a live local with the declared length.
    let validated = unsafe { sk_ffi_validate_descriptor(&descriptor) }.expect("valid descriptor");
    let matrix = validated.matrix_f64().expect("a 2×3 f64 matrix");
    assert_eq!(matrix.dim(), (2, 3));
    assert_eq!(matrix[[1, 2]], 6.0);
}

/// The length check runs before any element is read.
#[test]
fn boundary_rejects_length_mismatch_before_reading() {
    let values = [1.0_f64, 2.0];
    let shape = [2_i64, 2];
    let descriptor = SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 2,
        shape: shape.as_ptr(),
        strides: ptr::null(),
        data: values.as_ptr() as *const std::ffi::c_void,
        len: 3,
    };
    // SAFETY: the descriptor itself is live; the declared buffer is short on
    // purpose and must be rejected before it is read.
    let error =
        unsafe { sk_ffi_validate_descriptor(&descriptor) }.expect_err("3 elements cannot fill 2×2");
    assert!(matches!(
        error,
        crate::error_kind::SKFfiError::Array(SKFfiArrayError::LengthMismatch { .. })
    ));
}

/// Null descriptors and pointers are rejected.
#[test]
fn boundary_rejects_null_pointers() {
    // SAFETY: a null descriptor pointer is exactly the malformed input tested.
    assert!(unsafe { sk_ffi_validate_descriptor(ptr::null()) }.is_err());
    let descriptor = SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 2,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    };
    // SAFETY: the descriptor is live; its inner pointers are null on purpose.
    assert!(unsafe { sk_ffi_validate_descriptor(&descriptor) }.is_err());
}

/// Publishing an output makes its buffers owned by the caller and releasing
/// zeroes the descriptor.
#[test]
fn boundary_publish_and_release_output_round_trip() {
    let output = SKFfiOwnedArray::from_f64_vector(ndarray::array![1.0, 2.0, 3.0]);
    // SAFETY: publish transfers ownership; release receives it back exactly once.
    unsafe {
        let mut descriptor = sk_ffi_publish_output(output).expect("publishing succeeds");
        assert_eq!(descriptor.dtype, SKFfiDataType::F64 as i32);
        assert_eq!(descriptor.ndim, 1);
        assert_eq!(descriptor.len, 3);
        let values = std::slice::from_raw_parts(descriptor.data as *const f64, 3);
        assert_eq!(values, &[1.0, 2.0, 3.0]);
        sk_ffi_release_output(&mut descriptor).expect("release succeeds");
        assert!(descriptor.data.is_null());
        assert!(descriptor.shape.is_null());
        assert_eq!(descriptor.len, 0);
        sk_ffi_release_output(&mut descriptor).expect("a second release is a no-op");
    }
}
