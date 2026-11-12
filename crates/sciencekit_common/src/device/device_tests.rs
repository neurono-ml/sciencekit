//! Tests for the device vocabulary (spec `gpu-compute-contracts`).

use crate::observability::SKBackendKind;

use super::{
    SK_HOST_FALLBACK_BACKEND, SKDeviceCapabilities, SKDeviceInfo, SKFallbackReason,
    lock_device_inventory_for_test, sk_clear_device_inventory, sk_register_device_inventory,
    sk_registered_device_inventory, sk_resolve_operation_backend,
};

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

/// The process-wide device inventory starts empty and round-trips registration.
#[test]
fn device_inventory_starts_empty_and_round_trips_registration() {
    let _guard = lock_device_inventory_for_test();
    assert!(sk_registered_device_inventory().is_empty());

    let device = SKDeviceInfo::new(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1024,
        SKDeviceCapabilities::new(true, true, false),
    );
    sk_register_device_inventory(vec![device.clone()]);
    assert_eq!(sk_registered_device_inventory(), vec![device]);

    sk_clear_device_inventory();
    assert!(sk_registered_device_inventory().is_empty());
}

/// Operation backend resolution keeps host kinds and resolves device kinds via the inventory.
#[test]
fn operation_backend_resolution_consults_the_registered_inventory() {
    let _guard = lock_device_inventory_for_test();
    assert_eq!(
        sk_resolve_operation_backend(SKBackendKind::MatrixMultiply),
        (SKBackendKind::MatrixMultiply, None)
    );
    assert_eq!(
        sk_resolve_operation_backend(SKBackendKind::OpenCl),
        (
            SK_HOST_FALLBACK_BACKEND,
            Some(SKFallbackReason::NoDeviceAvailable)
        )
    );

    sk_register_device_inventory(vec![SKDeviceInfo::new(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1024,
        SKDeviceCapabilities::new(true, true, false),
    )]);
    assert_eq!(
        sk_resolve_operation_backend(SKBackendKind::OpenCl),
        (SKBackendKind::OpenCl, None)
    );
    assert_eq!(
        sk_resolve_operation_backend(SKBackendKind::Cuda),
        (
            SK_HOST_FALLBACK_BACKEND,
            Some(SKFallbackReason::NoDeviceAvailable)
        )
    );
}
