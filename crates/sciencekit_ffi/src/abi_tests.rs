//! Tests for the version and feature query (spec `ffi-abi`).

use super::{SK_FFI_ABI_MAJOR, SK_FFI_ABI_MINOR, SK_FFI_ABI_PATCH, SK_FFI_FEATURES};
use crate::boundary::{sk_ffi_features, sk_ffi_status_name, sk_ffi_version};
use crate::status::SKFfiStatus;

/// The version query fills every out-pointer and matches the constants.
#[test]
fn version_query_reports_the_abi_version() {
    let (mut major, mut minor, mut patch) = (-1, -1, -1);
    assert_eq!(
        sk_ffi_version(&mut major, &mut minor, &mut patch),
        SKFfiStatus::Ok
    );
    assert_eq!(
        (major, minor, patch),
        (SK_FFI_ABI_MAJOR, SK_FFI_ABI_MINOR, SK_FFI_ABI_PATCH)
    );
}

/// Null version out-pointers are rejected rather than written.
#[test]
fn version_query_rejects_null_pointers() {
    let mut value = 0_i32;
    assert_eq!(
        sk_ffi_version(std::ptr::null_mut(), &mut value, &mut value),
        SKFfiStatus::InvalidArgument
    );
}

/// The feature bitmask advertises the modeling, persistence and dtype support.
#[test]
fn version_features_report_the_supported_surface() {
    assert_eq!(sk_ffi_features(), SK_FFI_FEATURES);
    for feature in [
        super::SK_FFI_FEATURE_LINEAR_REGRESSION,
        super::SK_FFI_FEATURE_STANDARD_SCALER,
        super::SK_FFI_FEATURE_MODEL_PERSISTENCE,
        super::SK_FFI_FEATURE_F32,
        super::SK_FFI_FEATURE_F64,
    ] {
        assert!(SK_FFI_FEATURES & feature == feature);
    }
}

/// The hand-maintained C header declares every exported symbol.
#[test]
fn version_header_declares_every_entry_point() {
    let header = include_str!("../include/sciencekit.h");
    for entry_point in super::SK_FFI_ENTRY_POINTS {
        assert!(
            header.contains(entry_point),
            "`sciencekit.h` is missing the `{entry_point}` declaration"
        );
    }
    assert!(header.contains("typedef struct SKFfiArray"));
    assert!(header.contains("SK_FFI_STATUS_INVALID_HANDLE"));
}

/// Every status has a stable display name.
#[test]
fn version_status_names_are_stable() {
    let name = sk_ffi_status_name(SKFfiStatus::Shape as i32);
    // SAFETY: the function returns a static, NUL-terminated string.
    let name = unsafe { std::ffi::CStr::from_ptr(name) };
    assert_eq!(name.to_str().unwrap(), "shape");
    // SAFETY: the same contract applies to the unknown-status fallback.
    let unknown = unsafe { std::ffi::CStr::from_ptr(sk_ffi_status_name(99)) };
    assert_eq!(unknown.to_str().unwrap(), "unknown");
}
