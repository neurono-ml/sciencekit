//! Tests for the honest CPU fallback of the OpenCL backend (spec
//! `opencl-backend`, requirement *CPU fallback*).
//!
//! The fallback is tested in both configurations: the feature-off build must
//! call no OpenCL symbol and the feature-on build must report a missing device
//! the same way, so planning always receives a reason instead of a panic.

use ndarray::array;
use sciencekit_common::{SKError, SKFallbackReason, SKGpuKernel};

use super::SKOpenClBackend;
use crate::backend::{SKComputeBackend, SKDeviceDiscovery};

fn unavailable_backend() -> SKOpenClBackend {
    SKOpenClBackend::builder()
        .device_index(usize::MAX)
        .build_or_fallback()
}

/// A backend without a device reports the no-device reason.
#[test]
fn a_backend_without_a_device_records_the_fallback_reason() {
    let backend = unavailable_backend();
    assert!(!backend.is_available());
    assert_eq!(
        backend.fallback_reason(),
        Some(SKFallbackReason::NoDeviceAvailable)
    );
    let reason = backend
        .availability()
        .expect_err("an unavailable backend must not report success");
    assert!(matches!(reason, SKError::DeviceUnavailable { .. }));
}

/// Every contract operation on a fallback backend returns a structured error.
#[test]
fn fallback_operations_report_structured_errors() {
    let backend = unavailable_backend();
    let host = array![[1.0_f64, 2.0], [3.0, 4.0]];

    let upload = backend.upload(host.view()).expect_err("upload must fail");
    assert!(matches!(upload, SKError::DeviceUnavailable { .. }));

    let buffer = crate::SKDeviceBufferBuilder::new(sciencekit_common::SKBackendKind::OpenCl, 0)
        .shape(2, 2)
        .values(vec![1.0, 2.0, 3.0, 4.0])
        .build()
        .expect("a staging buffer");
    let download = backend.download(&buffer).expect_err("download must fail");
    assert!(matches!(download, SKError::DeviceUnavailable { .. }));

    let dispatch = backend
        .dispatch(SKGpuKernel::Gemm, &[&buffer, &buffer])
        .expect_err("dispatch must fail");
    assert!(matches!(dispatch, SKError::DeviceUnavailable { .. }));
}

/// Discovery of a fallback backend never invents a device.
#[test]
fn fallback_discovery_matches_the_enumeration() {
    let backend = unavailable_backend();
    let discovered = backend.discover_devices().expect("discovery must not fail");
    assert_eq!(discovered, crate::opencl::sk_opencl_devices().unwrap());
    if cfg!(feature = "opencl") {
        assert!(!discovered.is_empty(), "this host reports an OpenCL device");
    } else {
        assert!(discovered.is_empty());
    }
}

/// The strict builder reports why it could not open a backend.
#[test]
fn the_strict_builder_reports_the_reason() {
    let error = SKOpenClBackend::builder()
        .device_index(usize::MAX)
        .build()
        .expect_err("no device can serve an impossible index");
    match error {
        SKError::DeviceUnavailable { backend, reason } => {
            assert_eq!(backend, "opencl");
            assert!(!reason.is_empty());
        }
        other => panic!("unexpected error: {other}"),
    }
}

/// A real device serves the contract surface, or the feature-off build is
/// checked instead. On this host the branch always executes the device path.
#[cfg(feature = "opencl")]
#[test]
fn a_real_device_serves_the_contract_surface() {
    let backend = SKOpenClBackend::builder().build_or_fallback();
    assert!(
        backend.is_available(),
        "this host reports an OpenCL device: {}",
        backend.availability().unwrap_err()
    );
    let device = backend.device().expect("available device");
    println!(
        "[opencl] contract surface runs on device {}: `{}`",
        device.index, device.name
    );

    let host = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let uploaded = backend.upload(host.view()).expect("upload");
    assert_eq!(
        uploaded.backend_kind(),
        sciencekit_common::SKBackendKind::OpenCl
    );
    assert_eq!(backend.download(&uploaded).expect("download"), host);

    let identity = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let left = backend.upload(host.view()).expect("upload left");
    let right = backend.upload(identity.view()).expect("upload right");
    let product = backend
        .dispatch(SKGpuKernel::Gemm, &[&left, &right])
        .expect("gemm dispatch");
    assert_eq!(backend.download(&product).expect("download"), host);
    assert!(backend.transfer_report().total_bytes() > 0);
}
