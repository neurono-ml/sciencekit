//! Tests for the OpenCL operation span (spec `opencl-backend`, decision 5).
//!
//! Every operation records the configured backend kind, the operation name, the
//! device index and the bytes moved. The test captures the span a consumer's
//! subscriber observes and asserts those fields, so a regression to a hardcoded
//! host kind or a missing transfer count fails here.

use ndarray::Array2;
use tracing_subscriber::prelude::*;

use super::SKOpenClBackend;
use super::span_recorder::RecordingLayer;
use crate::opencl::SKOpenClDistanceMetric;

/// A pairwise dispatch records `opencl`, the operation, the device and bytes.
#[test]
fn a_kernel_operation_records_backend_device_and_bytes() {
    let backend = SKOpenClBackend::builder().build_or_fallback();
    if !backend.is_available() {
        println!("[opencl] no device available; observability test skips");
        return;
    }
    let device = backend.device().unwrap();
    println!(
        "[opencl] observability test runs on device {}: `{}`",
        device.index, device.name
    );
    let left = Array2::from_shape_fn((6, 5), |(row, column)| (row as f64) - (column as f64));
    let right = Array2::from_shape_fn((7, 5), |(row, column)| (row as f64) * 0.5 + (column as f64));

    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        backend
            .pairwise_distances(SKOpenClDistanceMetric::Euclidean, left.view(), right.view())
            .expect("pairwise dispatch on the real device");
    });

    let spans = recording.spans();
    let (_, fields) = spans
        .iter()
        .find(|(name, _)| name == "opencl_operation")
        .expect("the operation span must be recorded");
    assert_eq!(fields.get("backend").map(String::as_str), Some("opencl"));
    assert_eq!(
        fields.get("operation").map(String::as_str),
        Some("pairwise_distance")
    );
    assert_eq!(
        fields.get("device_index").map(String::as_str),
        Some(device.index.to_string().as_str())
    );
    let bytes: u64 = fields
        .get("transfer_bytes")
        .expect("transfer bytes must be recorded")
        .parse()
        .expect("transfer bytes must be an integer");
    assert!(bytes > 0, "a real dispatch moves bytes");
}
