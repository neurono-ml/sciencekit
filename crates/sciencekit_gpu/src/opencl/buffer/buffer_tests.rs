//! Tests for device buffers and host↔device transfers (spec `opencl-backend`,
//! requirement *OpenCL buffers*).

use ndarray::{Array2, array};
use sciencekit_common::SKError;

use super::{SKOpenClAllocation, download_buffer, upload_view};
use crate::opencl::SKOpenClBackend;

fn backend() -> Option<SKOpenClBackend> {
    let backend = SKOpenClBackend::builder().build_or_fallback();
    if !backend.is_available() {
        println!(
            "[opencl] no device available; buffer tests skip: {}",
            backend.fallback_reason().is_some()
        );
        return None;
    }
    let device = backend.device().expect("available backend has a device");
    println!(
        "[opencl] buffer tests run on device {}: `{}`",
        device.index, device.name
    );
    Some(backend)
}

/// A host matrix uploaded and downloaded again is unchanged, in both precisions.
#[test]
fn buffers_round_trip_without_loss() {
    let Some(backend) = backend() else { return };
    let session = backend.session().expect("a device session");

    let single: Array2<f32> = Array2::from_shape_fn((5, 7), |(row, column)| {
        ((row * 7 + column * 3) as f32) * 0.25 - 3.0
    });
    let single_resident = upload_view(session, single.view()).expect("upload f32");
    assert_eq!(single_resident.shape(), (5, 7));
    assert_eq!(single_resident.scalar_type(), "f32");
    assert_eq!(single_resident.element_count(), 35);
    let single_back = download_buffer(session, &single_resident).expect("download f32");
    assert_eq!(single_back, single);

    let double: Array2<f64> = Array2::from_shape_fn((9, 4), |(row, column)| {
        ((row as f64) - 4.0) * 1.5 + (column as f64) * 0.125
    });
    let double_resident = upload_view(session, double.view()).expect("upload f64");
    let double_back = download_buffer(session, &double_resident).expect("download f64");
    assert_eq!(double_back, double);
}

/// A single-row and a single-column matrix still round-trip (no shape guessing).
#[test]
fn degenerate_shapes_round_trip() {
    let Some(backend) = backend() else { return };
    let session = backend.session().expect("a device session");

    let row: Array2<f64> = array![[1.0, -2.0, 3.5, 0.0]];
    let row_back =
        download_buffer(session, &upload_view(session, row.view()).unwrap()).expect("row");
    assert_eq!(row_back, row);

    let column: Array2<f64> = array![[1.0], [2.0], [3.5]];
    let column_back =
        download_buffer(session, &upload_view(session, column.view()).unwrap()).expect("column");
    assert_eq!(column_back, column);
}

/// A buffer that cannot fit in device memory reports a structured error instead
/// of aborting the process.
#[test]
fn an_oversized_allocation_reports_a_structured_error() {
    let Some(backend) = backend() else { return };
    let session = backend.session().expect("a device session");
    let error = SKOpenClAllocation::allocate(session, 1 << 42, "f64")
        .expect_err("a 32 TiB allocation cannot succeed");
    assert!(matches!(error, SKError::DeviceUnavailable { .. }));

    let capacity = SKOpenClAllocation::allocate(session, 4, "f32").expect("a tiny allocation");
    let mut capacity = capacity;
    let error = capacity
        .write_from_host(session, &[1.0_f32; 8])
        .expect_err("the payload exceeds the capacity");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The accounting report counts the bytes the backend uploaded and downloaded.
#[test]
fn transfers_are_counted() {
    let Some(backend) = backend() else { return };
    let host: Array2<f64> =
        Array2::from_shape_fn((6, 5), |(row, column)| (row as f64) - (column as f64));
    let resident = backend.upload_resident(host.view()).expect("upload");
    let back = backend.download_resident(&resident).expect("download");
    assert_eq!(back, host);

    let report = backend.transfer_report();
    assert_eq!(report.upload_bytes, 30 * 8);
    assert_eq!(report.download_bytes, 30 * 8);
    assert!(report.total_bytes() >= 30 * 8 * 2);
}
