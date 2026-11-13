//! Concurrency tests for the OpenCL backend (spec `opencl-backend`,
//! requirement *Concurrency and parity*).
//!
//! One backend is shared across threads; the session's command queue and
//! program cache are `Send + Sync`, so concurrent dispatches must return exactly
//! the sequential result.

use std::sync::Arc;
use std::thread;

use ndarray::{Array2, array};
use sciencekit_common::SKFloat;

use super::SKOpenClBackend;
use crate::opencl::{SKOpenClBuffer, SKOpenClDistanceMetric};

const THREADS: usize = 8;

fn assert_send_and_sync<T: Send + Sync + ?Sized>() {}

fn backend() -> Option<Arc<SKOpenClBackend>> {
    let backend = SKOpenClBackend::builder().build_or_fallback();
    if !backend.is_available() {
        println!("[opencl] no device available; concurrency tests skip");
        return None;
    }
    let device = backend.device().unwrap();
    println!(
        "[opencl] concurrency tests run on device {}: `{}`",
        device.index, device.name
    );
    Some(Arc::new(backend))
}

fn data(seed: f64) -> Array2<f64> {
    Array2::from_shape_fn((64, 9), |(row, column)| {
        seed + (row as f64) * 0.125 - (column as f64) * 0.0625
    })
}

/// The backend and its buffers are safe to share and send.
#[test]
fn backend_types_are_send_and_sync() {
    assert_send_and_sync::<SKOpenClBackend>();
    assert_send_and_sync::<SKOpenClBuffer<f32>>();
    assert_send_and_sync::<SKOpenClBuffer<f64>>();
}

/// Concurrent pairwise dispatches agree with the sequential result.
#[test]
fn concurrent_pairwise_dispatches_agree() {
    let Some(backend) = backend() else { return };
    let sequential = backend
        .pairwise_distances(
            SKOpenClDistanceMetric::Euclidean,
            data(0.0).view(),
            data(1.0).view(),
        )
        .expect("sequential dispatch");

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let backend = Arc::clone(&backend);
            thread::spawn(move || {
                backend
                    .pairwise_distances(
                        SKOpenClDistanceMetric::Euclidean,
                        data(0.0).view(),
                        data(1.0).view(),
                    )
                    .expect("concurrent dispatch")
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("thread must not panic");
        assert_eq!(result, sequential);
    }
}

/// Concurrent uploads and downloads over the same backend agree.
#[test]
fn concurrent_transfers_agree() {
    let Some(backend) = backend() else { return };
    let host: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let backend = Arc::clone(&backend);
            let host = host.clone();
            thread::spawn(move || {
                let resident = backend.upload_resident(host.view()).expect("upload");
                backend.download_resident(&resident).expect("download")
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().expect("thread must not panic"), host);
    }
}

/// The scalar-size helper the kernels use stays honest for both precisions.
#[test]
fn element_sizes_are_reported_by_scalar_name() {
    assert_eq!(std::mem::size_of::<f32>(), 4);
    assert_eq!(std::mem::size_of::<f64>(), 8);
    assert_eq!(<f32 as SKFloat>::SCALAR_NAME, "f32");
    assert_eq!(<f64 as SKFloat>::SCALAR_NAME, "f64");
}
