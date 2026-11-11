//! Tests for device buffers and host↔device transfers (spec
//! `gpu-compute-contracts`).

use ndarray::array;
use sciencekit_common::SKBackendKind;

use crate::backend::mock_device::SKMockComputeBackend;
use crate::{
    SK_HOST_FALLBACK_BACKEND, SKComputeBackend, SKDeviceBuffer, SKDeviceBufferBuilder,
    SKDeviceBufferRef, SKDeviceCapabilities, SKDeviceContext, SKDeviceDiscovery, SKDeviceInfo,
    SKDevicePreference, SKDeviceTransferPolicy, SKFallbackReason, SKGpuKernel, sk_select_device,
};

fn backend() -> SKMockComputeBackend {
    SKMockComputeBackend::new(vec![SKMockComputeBackend::device(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1 << 20,
        sciencekit_common::SKDeviceCapabilities::new(true, true, true),
    )])
}

/// A host buffer uploaded and downloaded again keeps every value.
#[test]
fn a_host_buffer_round_trips_through_a_mock_device() {
    let host = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let backend = backend();

    let device_buffer = backend.upload(host.view()).expect("upload must succeed");
    let downloaded = backend
        .download(&device_buffer)
        .expect("download must succeed");

    assert_eq!(downloaded, host);
}

/// The buffer carries the device, shape and capacity it was allocated with.
#[test]
fn a_device_buffer_reports_its_allocation() {
    let host = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let buffer: SKDeviceBuffer<f64> = backend().upload(host.view()).expect("upload must succeed");

    assert_eq!(buffer.backend_kind(), SKBackendKind::OpenCl);
    assert_eq!(buffer.device_index(), 0);
    assert_eq!(buffer.shape(), (3, 2));
    assert_eq!(buffer.capacity(), 6);
    assert_eq!(buffer.scalar_type(), "f64");
}

/// Single-precision buffers round-trip as well.
#[test]
fn single_precision_buffers_round_trip() {
    let host = array![[0.5_f32, -1.5], [2.5, 3.5]];
    let backend = backend();
    let buffer = backend.upload(host.view()).expect("upload must succeed");
    let downloaded = backend.download(&buffer).expect("download must succeed");
    assert_eq!(downloaded, host);
}

/// The safe default transfer policy is an explicit copy, and backends declare theirs.
#[test]
fn the_default_transfer_policy_is_explicit() {
    assert_eq!(
        SKDeviceTransferPolicy::default(),
        SKDeviceTransferPolicy::Explicit
    );
    let explicit_backend = backend();
    assert_eq!(
        <SKMockComputeBackend as SKComputeBackend<f64>>::transfer_policy(&explicit_backend),
        SKDeviceTransferPolicy::Explicit
    );
    let mapped_backend =
        backend().with_transfer_policy(SKDeviceTransferPolicy::MappedWhenSupported);
    assert_eq!(
        <SKMockComputeBackend as SKComputeBackend<f64>>::transfer_policy(&mapped_backend),
        SKDeviceTransferPolicy::MappedWhenSupported
    );
}

/// Crate-root re-exports expose the complete public GPU contract surface.
#[test]
fn crate_root_re_exports_the_public_contract_surface() {
    let devices = backend().discover_devices().expect("discovery succeeds");
    let context = SKDeviceContext::new(SKBackendKind::OpenCl)
        .with_required_kernel(SKGpuKernel::Gemm)
        .with_inventory(devices);
    let selected: Result<SKDeviceInfo, SKFallbackReason> = sk_select_device(
        &context.device_inventory,
        SKDevicePreference::AnyDevice,
        None,
        Some("f64"),
    );
    let selected = selected.expect("one OpenCL device is present");
    let built = SKDeviceBufferBuilder::<f64>::new(selected.kind, selected.index)
        .shape(1, 1)
        .values(vec![1.0])
        .build()
        .expect("valid device buffer");
    let refs: &[SKDeviceBufferRef<'_, f64>] = &[&built];
    assert_eq!(refs.len(), 1);
    assert_eq!(SK_HOST_FALLBACK_BACKEND, SKBackendKind::Faer);
    assert!(SKDeviceCapabilities::new(true, true, true).supports_mapped_memory);
}
