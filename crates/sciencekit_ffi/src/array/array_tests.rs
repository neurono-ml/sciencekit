//! Tests for descriptor validation (spec `ffi-abi`).

use super::{
    SKFfiArrayShape, SKFfiDataType, sk_ffi_validate_contiguous_strides, sk_ffi_validate_length,
};

/// Dtype codes round-trip and report their element widths.
#[test]
fn array_dtype_codes_round_trip() {
    for (code, dtype) in [
        (0, SKFfiDataType::F32),
        (1, SKFfiDataType::F64),
        (2, SKFfiDataType::I64),
        (3, SKFfiDataType::U8),
    ] {
        assert_eq!(SKFfiDataType::from_code(code), Some(dtype));
        assert_eq!(dtype as i32, code);
        assert!(dtype.element_size() > 0);
    }
    assert_eq!(SKFfiDataType::from_code(9), None);
    assert!(SKFfiDataType::F64.is_floating());
    assert!(!SKFfiDataType::U8.is_floating());
}

/// A descriptor whose length disagrees with its shape is rejected before any
/// element is read.
#[test]
fn array_length_mismatch_is_rejected() {
    let shape = SKFfiArrayShape::new(&[3, 2]).expect("a positive shape is valid");
    let error = sk_ffi_validate_length(&shape, 5).expect_err("5 elements cannot fill 3×2");
    assert!(error.to_string().contains("expected 6"));
    assert!(error.to_string().contains("found 5"));
    assert_eq!(sk_ffi_validate_length(&shape, 6), Ok(6));
}

/// Zero and negative dimensions are rejected.
#[test]
fn array_rejects_empty_dimensions() {
    assert!(SKFfiArrayShape::new(&[0, 2]).is_err());
    assert!(SKFfiArrayShape::new(&[-1]).is_err());
    assert!(SKFfiArrayShape::new(&[]).is_err());
    assert!(SKFfiArrayShape::new(&[1, 2, 3]).is_err());
}

/// Null strides mean row-major; explicit strides must describe it exactly.
#[test]
fn array_accepts_only_contiguous_strides() {
    let shape = SKFfiArrayShape::new(&[3, 2]).expect("a positive shape is valid");
    assert_eq!(sk_ffi_validate_contiguous_strides(None, &shape), Ok(()));
    assert_eq!(
        sk_ffi_validate_contiguous_strides(Some(&[2, 1]), &shape),
        Ok(())
    );
    assert!(sk_ffi_validate_contiguous_strides(Some(&[1, 3]), &shape).is_err());
    assert!(sk_ffi_validate_contiguous_strides(Some(&[2]), &shape).is_err());
}

/// Rank helpers reject the wrong rank with a precise message.
#[test]
fn array_rank_helpers_reject_the_wrong_rank() {
    let matrix = SKFfiArrayShape::new(&[3, 2]).expect("valid");
    assert_eq!(matrix.matrix_shape(), Ok((3, 2)));
    assert!(matrix.vector_length().is_err());
    let vector = SKFfiArrayShape::new(&[4]).expect("valid");
    assert_eq!(vector.vector_length(), Ok(4));
    assert!(vector.matrix_shape().is_err());
}
