//! Validation tests for buffers, transfers and device compatibility (spec
//! `gpu-compute-contracts`).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKDeviceCapabilities, SKError};

use crate::backend::SKComputeBackend;
use crate::backend::mock_device::SKMockComputeBackend;
use crate::buffer::SKDeviceBufferBuilder;

fn backend_with(capabilities: SKDeviceCapabilities) -> SKMockComputeBackend {
    SKMockComputeBackend::new(vec![SKMockComputeBackend::device(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1 << 20,
        capabilities,
    )])
}

/// A device that does not execute the scalar type rejects the upload.
#[test]
fn an_unsupported_scalar_type_is_rejected() {
    let backend = backend_with(SKDeviceCapabilities::new(true, false, false));
    let host = array![[1.0_f64, 2.0]];

    let error = backend
        .upload(host.view())
        .expect_err("the mock device only supports f32");
    match error {
        SKError::UnsupportedScalarType {
            backend,
            scalar_type,
        } => {
            assert_eq!(backend, "mock");
            assert_eq!(scalar_type, "f64");
        }
        other => panic!("expected `UnsupportedScalarType`, got {other:?}"),
    }
}

/// An upload larger than the device's allocation capacity is a shape mismatch.
#[test]
fn an_upload_exceeding_the_device_capacity_returns_shape_mismatch() {
    let backend = backend_with(SKDeviceCapabilities::new(true, true, false)).limited_to_elements(3);
    let host = array![[1.0_f64, 2.0], [3.0, 4.0]];

    let error = backend
        .upload(host.view())
        .expect_err("four elements exceed the three-element capacity");
    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![3]);
            assert_eq!(found, vec![4]);
        }
        other => panic!("expected `ShapeMismatch`, got {other:?}"),
    }
}

/// A shape larger than the buffer capacity is rejected at construction.
#[test]
fn a_shape_exceeding_the_buffer_capacity_is_rejected() {
    let error = SKDeviceBufferBuilder::<f64>::new(SKBackendKind::OpenCl, 0)
        .shape(2, 2)
        .capacity(3)
        .values(vec![1.0, 2.0, 3.0, 4.0])
        .build()
        .expect_err("four elements exceed the three-element capacity");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A payload that does not match the declared shape is rejected.
#[test]
fn a_payload_that_does_not_match_its_shape_is_rejected() {
    let error = SKDeviceBufferBuilder::<f64>::new(SKBackendKind::OpenCl, 0)
        .shape(2, 2)
        .values(vec![1.0, 2.0, 3.0])
        .build()
        .expect_err("three values cannot fill a 2×2 buffer");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A transfer failure during upload is a structured error, never a panic.
#[test]
fn a_failed_upload_transfer_is_a_structured_error() {
    let backend = backend_with(SKDeviceCapabilities::new(true, true, false))
        .failing_transfers("device memory exhausted");
    let host = array![[1.0_f64, 2.0]];

    let error = backend
        .upload(host.view())
        .expect_err("the injected upload failure must surface");
    match error {
        SKError::DeviceTransferFailure { reason } => {
            assert!(reason.contains("device memory exhausted"));
        }
        other => panic!("expected `DeviceTransferFailure`, got {other:?}"),
    }
}

/// A transfer failure during download is a structured error, never a panic.
#[test]
fn a_failed_download_transfer_is_a_structured_error() {
    let healthy = backend_with(SKDeviceCapabilities::new(true, true, false));
    let host = array![[1.0_f64, 2.0]];
    let buffer = healthy.upload(host.view()).expect("upload must succeed");

    let failing =
        backend_with(SKDeviceCapabilities::new(true, true, false)).failing_transfers("device lost");
    let error = failing
        .download(&buffer)
        .expect_err("the injected download failure must surface");
    assert!(matches!(error, SKError::DeviceTransferFailure { .. }));
}

/// A backend with no device cannot accept an upload.
#[test]
fn uploading_to_a_backend_without_devices_reports_the_device_unavailable() {
    let backend = SKMockComputeBackend::without_devices();
    let host = array![[1.0_f64]];

    let error = backend
        .upload(host.view())
        .expect_err("no device is available");
    assert!(matches!(error, SKError::DeviceUnavailable { .. }));
}

/// A buffer from another backend cannot be downloaded through this one.
#[test]
fn downloading_a_buffer_from_another_backend_is_rejected() {
    let opencl = backend_with(SKDeviceCapabilities::new(true, true, false));
    let host = array![[1.0_f64]];
    let buffer = opencl.upload(host.view()).expect("upload must succeed");

    let cuda = SKMockComputeBackend::new(vec![SKMockComputeBackend::device(
        0,
        SKBackendKind::Cuda,
        "Mock CUDA 0",
        1 << 20,
        SKDeviceCapabilities::new(true, true, false),
    )]);
    let error = cuda
        .download(&buffer)
        .expect_err("the buffer belongs to the OpenCL backend");
    match error {
        SKError::DeviceUnavailable { backend, .. } => assert_eq!(backend, "opencl"),
        other => panic!("expected `DeviceUnavailable`, got {other:?}"),
    }
}

/// A host backend kind cannot own a device buffer.
#[test]
fn a_host_backend_cannot_own_a_device_buffer() {
    let error = SKDeviceBufferBuilder::<f64>::new(SKBackendKind::Faer, 0)
        .shape(1, 1)
        .values(vec![1.0])
        .build()
        .expect_err("a host backend must not own a device buffer");
    match error {
        SKError::DeviceUnavailable { backend, .. } => assert_eq!(backend, "faer"),
        other => panic!("expected `DeviceUnavailable`, got {other:?}"),
    }
}

/// A `(rows, columns)` product that overflows `usize` is rejected as a shape mismatch.
#[test]
fn an_overflowing_shape_returns_shape_mismatch() {
    let error = SKDeviceBufferBuilder::<f64>::new(SKBackendKind::OpenCl, 0)
        .shape(usize::MAX, 2)
        .build()
        .expect_err("usize::MAX * 2 overflows");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
