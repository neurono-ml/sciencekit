//! Tests for the honest CPU fallback of the OpenCL backend (spec
//! `opencl-backend`, requirement *CPU fallback*).
//!
//! The fallback is tested in both configurations: the feature-off build must
//! call no OpenCL symbol and the feature-on build must report a missing device
//! the same way, so planning always receives a reason instead of a panic.
//!
//! The tests never depend on the hardware of the host that runs them. An
//! unavailable backend is constructed explicitly with
//! [`SKOpenClBackend::unavailable`], so the same code path runs on a machine
//! with a GPU and on a machine without one; a test that only passes because a
//! device happens to be visible would hide the fallback defect from CI.

use std::collections::HashMap;

#[cfg(feature = "opencl")]
use ndarray::Array2;
use ndarray::array;
use sciencekit_common::{SKError, SKFallbackReason, SKGpuKernel};
use tracing_subscriber::prelude::*;

use super::SKOpenClBackend;
use super::span_recorder::RecordingLayer;
use crate::backend::{SKComputeBackend, SKDeviceDiscovery};
use crate::buffer::SKDeviceTransferPolicy;

/// A backend that is unavailable by construction, whatever the host exposes.
fn unavailable_backend() -> SKOpenClBackend {
    SKOpenClBackend::unavailable(
        "the test forces the unavailable path".to_string(),
        SKDeviceTransferPolicy::Explicit,
    )
}

/// Capture the spans the operation opens under a private subscriber.
fn capture_spans(run: impl FnOnce()) -> Vec<(String, HashMap<String, String>)> {
    let recorder = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    tracing::subscriber::with_default(subscriber, run);
    recorder.spans()
}

/// Assert one operation opened exactly one span carrying the host fallback.
fn assert_single_fallback_span(spans: &[(String, HashMap<String, String>)], operation: &str) {
    assert_eq!(
        spans.len(),
        1,
        "`{operation}` must open its span exactly once"
    );
    let (name, fields) = &spans[0];
    assert_eq!(name, "opencl_operation");
    assert_eq!(fields.get("operation").map(String::as_str), Some(operation));
    assert_eq!(
        fields.get("backend").map(String::as_str),
        Some("faer"),
        "a failed device operation must report the host fallback that serves it"
    );
    assert_eq!(
        fields.get("fallback_reason").map(String::as_str),
        Some("no-device-available")
    );
    assert!(
        !fields.contains_key("device_index"),
        "no device ran the operation"
    );
    assert!(
        !fields.contains_key("transfer_bytes"),
        "no bytes crossed the bus"
    );
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

/// Discovery of an unavailable backend never invents a device: it reports the
/// same enumeration the host exposes, whatever that is.
#[test]
fn fallback_discovery_matches_the_enumeration() {
    let backend = unavailable_backend();
    let mut discovered = Vec::new();
    let spans = capture_spans(|| {
        discovered = backend.discover_devices().expect("discovery must not fail");
    });
    assert_eq!(discovered, crate::opencl::sk_opencl_devices().unwrap());
    assert_single_fallback_span(&spans, "discover_devices");
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

/// A real device serves the contract surface; a host without one skips the
/// device branch instead of pretending the test proved anything about it.
#[cfg(feature = "opencl")]
#[test]
fn a_real_device_serves_the_contract_surface() {
    let backend = SKOpenClBackend::builder().build_or_fallback();
    if !backend.is_available() {
        println!("[opencl] no device available; the device path skips");
        return;
    }
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

/// An unavailable operation never claims device work on its span: it records the
/// host fallback and why, so a trace cannot mistake a failed device attempt for
/// a kernel that ran on OpenCL.
///
/// The backend is unavailable by construction, so this assertion is identical on
/// a host with a GPU and on a host without one; it does not depend on the
/// enumeration the runner happens to expose.
#[test]
fn an_unavailable_operation_records_the_host_fallback_and_reason() {
    let backend = unavailable_backend();
    let host = array![[1.0_f64, 2.0], [3.0, 4.0]];

    let mut upload_error = None;
    let spans = capture_spans(|| {
        upload_error = Some(
            backend
                .upload(host.view())
                .expect_err("an unavailable backend must reject the upload"),
        );
    });
    assert!(matches!(
        upload_error.expect("the upload ran"),
        SKError::DeviceUnavailable { .. }
    ));
    assert_single_fallback_span(&spans, "upload");
}

/// Every public typed operation opens the same honest fallback span, exactly
/// once, even though no session can serve it.
#[cfg(feature = "opencl")]
#[test]
fn every_typed_operation_records_the_host_fallback_once() {
    use crate::opencl::SKOpenClDistanceMetric;

    /// Run one operation under a private subscriber and assert its span.
    macro_rules! assert_operation_span {
        ($backend:expr, $operation:literal, $body:expr) => {{
            let recorder = RecordingLayer::default();
            let subscriber = tracing_subscriber::registry().with(recorder.clone());
            tracing::subscriber::with_default(subscriber, || {
                $body;
            });
            assert_single_fallback_span(&recorder.spans(), $operation);
        }};
    }

    let backend = unavailable_backend();
    let left = Array2::from_shape_fn((4, 3), |(row, column)| (row + column) as f64);
    let right = Array2::from_shape_fn((5, 3), |(row, column)| (row * 2 + column) as f64);
    let square = Array2::from_shape_fn((3, 3), |(row, column)| (row + column) as f64);

    assert_operation_span!(backend, "upload_resident", {
        let _ = backend.upload_resident(left.view());
    });
    assert_operation_span!(backend, "pairwise_distance", {
        let _ = backend.pairwise_distances(
            SKOpenClDistanceMetric::Euclidean,
            left.view(),
            right.view(),
        );
    });
    assert_operation_span!(backend, "gemm", {
        let _ = backend.gemm(left.view(), square.view());
    });
    assert_operation_span!(backend, "elementwise_transform", {
        let _ = backend.transform(left.view(), 2.0_f64, 1.0);
    });
    assert_operation_span!(backend, "elementwise_combine", {
        let _ = backend.combine(left.view(), left.view());
    });
    assert_operation_span!(backend, "tree_predict", {
        let _ = backend.predict_trees(left.view(), left.view());
    });
}
