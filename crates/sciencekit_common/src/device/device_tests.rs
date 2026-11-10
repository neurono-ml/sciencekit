//! Tests for the device vocabulary (spec `gpu-compute-contracts`).

use crate::observability::SKBackendKind;

use super::{SKDeviceCapabilities, SKDeviceInfo};

/// A described device reports its identity, memory and capabilities verbatim.
#[test]
fn device_info_reports_identity_memory_and_capabilities() {
    let device = SKDeviceInfo::new(
        2,
        SKBackendKind::Cuda,
        "Mock CUDA Device",
        16 * 1024 * 1024 * 1024,
        SKDeviceCapabilities::new(true, false, true),
    );
    assert_eq!(device.index, 2);
    assert_eq!(device.kind, SKBackendKind::Cuda);
    assert_eq!(device.name, "Mock CUDA Device");
    assert_eq!(device.memory_bytes, 16 * 1024 * 1024 * 1024);
    assert!(device.is_device());
}

/// Scalar-type support follows the capability flags and never guesses.
#[test]
fn scalar_type_support_follows_the_capability_flags() {
    let single_precision = SKDeviceCapabilities::new(true, false, false);
    assert!(single_precision.supports_scalar_type("f32"));
    assert!(!single_precision.supports_scalar_type("f64"));
    assert!(!single_precision.supports_scalar_type("f16"));
}

/// A default capability summary supports nothing.
#[test]
fn the_default_capabilities_support_nothing() {
    let capabilities = SKDeviceCapabilities::default();
    assert!(!capabilities.supports_f32);
    assert!(!capabilities.supports_f64);
    assert!(!capabilities.supports_mapped_memory);
    assert!(!capabilities.supports_scalar_type("f32"));
}
