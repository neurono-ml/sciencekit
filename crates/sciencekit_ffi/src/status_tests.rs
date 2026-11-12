//! Tests for the stable status codes (spec `ffi-abi`).

use super::{SKFfiStatus, sk_ffi_status_for_error};
use sciencekit_common::SKError;

/// Every code keeps the integer value bindings were compiled against.
#[test]
fn status_codes_are_stable_and_round_trip() {
    for (code, expected) in [
        (0, SKFfiStatus::Ok),
        (1, SKFfiStatus::InvalidHandle),
        (2, SKFfiStatus::Shape),
        (3, SKFfiStatus::InvalidArgument),
        (4, SKFfiStatus::Unsupported),
        (5, SKFfiStatus::NotFitted),
        (6, SKFfiStatus::Io),
        (7, SKFfiStatus::Internal),
    ] {
        assert_eq!(SKFfiStatus::from_code(code), Some(expected));
        assert_eq!(expected as i32, code);
        assert!(!expected.name().is_empty());
    }
    assert_eq!(SKFfiStatus::from_code(99), None);
}

/// Shape mismatches map to the dedicated shape status.
#[test]
fn status_maps_shape_errors_to_shape() {
    let error = SKError::shape_mismatch_2d(3, 2, 2, 3);
    assert_eq!(sk_ffi_status_for_error(&error), SKFfiStatus::Shape);
}

/// Unsupported representations and operations map to unsupported.
#[test]
fn status_maps_unsupported_failures_to_unsupported() {
    let representation = SKError::UnsupportedRepresentation {
        representation: "csr",
        suggestion: "dense only",
    };
    assert_eq!(
        sk_ffi_status_for_error(&representation),
        SKFfiStatus::Unsupported
    );
}

/// I/O failures keep their dedicated status so bindings can retry differently.
#[test]
fn status_maps_io_errors_to_io() {
    let error = SKError::Io(std::io::Error::other("disk"));
    assert_eq!(sk_ffi_status_for_error(&error), SKFfiStatus::Io);
}

/// Unavailable devices and unsupported scalar types map to `Unsupported`.
#[test]
fn status_maps_device_capability_errors_to_unsupported() {
    let unavailable = SKError::DeviceUnavailable {
        backend: "opencl",
        reason: "no ICD found".to_string(),
    };
    let unsupported_scalar = SKError::UnsupportedScalarType {
        backend: "cuda",
        scalar_type: "f64",
    };
    assert_eq!(
        sk_ffi_status_for_error(&unavailable),
        SKFfiStatus::Unsupported
    );
    assert_eq!(
        sk_ffi_status_for_error(&unsupported_scalar),
        SKFfiStatus::Unsupported
    );
}

/// A host↔device transfer failure after selection maps to `Internal`.
#[test]
fn status_maps_device_transfer_failure_to_internal() {
    let transfer = SKError::DeviceTransferFailure {
        reason: "DMA timeout".to_string(),
    };
    assert_eq!(sk_ffi_status_for_error(&transfer), SKFfiStatus::Internal);
}
