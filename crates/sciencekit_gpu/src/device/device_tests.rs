//! Tests for device discovery over mock backends (spec
//! `gpu-compute-contracts`).

use sciencekit_common::{SKBackendKind, SKDeviceCapabilities, SKError};

use crate::backend::SKDeviceDiscovery;
use crate::backend::mock_device::SKMockComputeBackend;

fn capabilities() -> SKDeviceCapabilities {
    SKDeviceCapabilities::new(true, true, false)
}

/// Enumeration preserves the backend's inventory order.
#[test]
fn enumeration_returns_devices_in_inventory_order() {
    let backend = SKMockComputeBackend::new(vec![
        SKMockComputeBackend::device(
            0,
            SKBackendKind::OpenCl,
            "Mock OpenCL 0",
            1024,
            capabilities(),
        ),
        SKMockComputeBackend::device(
            1,
            SKBackendKind::OpenCl,
            "Mock OpenCL 1",
            2048,
            capabilities(),
        ),
        SKMockComputeBackend::device(0, SKBackendKind::Cuda, "Mock CUDA 0", 4096, capabilities()),
    ]);

    let devices = backend.discover_devices().expect("discovery must succeed");
    let names: Vec<&str> = devices.iter().map(|device| device.name.as_str()).collect();
    assert_eq!(names, vec!["Mock OpenCL 0", "Mock OpenCL 1", "Mock CUDA 0"]);
    let indexes: Vec<usize> = devices.iter().map(|device| device.index).collect();
    assert_eq!(indexes, vec![0, 1, 0]);
}

/// A backend with no device enumerates an empty list instead of failing.
#[test]
fn enumeration_without_devices_returns_an_empty_list() {
    let backend = SKMockComputeBackend::without_devices();
    let devices = backend.discover_devices().expect("discovery must succeed");
    assert!(devices.is_empty());
}

/// Device descriptions carry the kind, memory and capabilities verbatim.
#[test]
fn enumerated_devices_describe_their_capabilities() {
    let backend = SKMockComputeBackend::new(vec![SKMockComputeBackend::device(
        0,
        SKBackendKind::Rocm,
        "Mock ROCm 0",
        8192,
        SKDeviceCapabilities::new(true, false, true),
    )]);

    let devices = backend.discover_devices().expect("discovery must succeed");
    let device = &devices[0];
    assert_eq!(device.kind, SKBackendKind::Rocm);
    assert_eq!(device.memory_bytes, 8192);
    assert!(device.supports_scalar_type("f32"));
    assert!(!device.supports_scalar_type("f64"));
    assert!(device.capabilities.supports_mapped_memory);
}

/// A discovery failure surfaces as a structured, non-panicking error.
#[test]
fn discovery_failure_is_a_structured_error() {
    let backend = SKMockComputeBackend::failing_discovery("device node denied");
    let error = backend
        .discover_devices()
        .expect_err("the mock failure must surface");
    match error {
        SKError::DeviceUnavailable { backend, reason } => {
            assert_eq!(backend, "mock");
            assert!(reason.contains("device node denied"));
        }
        other => panic!("expected `DeviceUnavailable`, got {other:?}"),
    }
}
