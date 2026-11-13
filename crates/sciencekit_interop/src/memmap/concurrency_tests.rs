//! TDD suite for concurrent readers and thread-safety contracts (task 4.1).

use std::sync::Arc;

use sciencekit_common::SKMappableSource;

use super::arrow_ipc_source::SKArrowIpcMappableSource;
use super::descriptor::{SKRawBinaryDescriptor, SKRawBinaryDtype};
use super::numpy_source::SKNumpyMappableSource;
use super::raw_binary_source::SKRawBinaryMappableSource;
use super::test_fixtures::{sk_f64_bytes, sk_write_bytes, sk_write_ipc_f64, sk_write_npy};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn mapped_sources_are_send_and_sync() {
    assert_send_sync::<SKRawBinaryMappableSource<f64>>();
    assert_send_sync::<SKNumpyMappableSource<f64>>();
    assert_send_sync::<SKArrowIpcMappableSource<f64>>();
}

#[test]
fn eight_threads_read_disjoint_rows_from_a_raw_file() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.raw");
    let rows = 8_000;
    let values: Vec<f64> = (0..rows * 2).map(|index| index as f64).collect();
    sk_write_bytes(&path, &sk_f64_bytes(&values));
    let source = Arc::new(
        SKRawBinaryMappableSource::<f64>::open(
            &path,
            SKRawBinaryDescriptor::new(rows, 2, SKRawBinaryDtype::Float64),
        )
        .expect("an open source"),
    );
    std::thread::scope(|scope| {
        for thread_index in 0..8 {
            let source = Arc::clone(&source);
            scope.spawn(move || {
                for step in 0..1_000 {
                    let row = thread_index + step * 8;
                    let view = source.row(row).expect("a valid row");
                    assert_eq!(view[0], (row * 2) as f64);
                    assert_eq!(view[1], (row * 2 + 1) as f64);
                }
            });
        }
    });
}

#[test]
fn eight_threads_read_disjoint_rows_from_a_numpy_file() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.npy");
    let rows = 800;
    let values: Vec<f64> = (0..rows * 2).map(|index| index as f64).collect();
    sk_write_npy(&path, "<f8", rows, 2, false, true, &sk_f64_bytes(&values));
    let source = Arc::new(SKNumpyMappableSource::<f64>::open(&path).expect("an open source"));
    std::thread::scope(|scope| {
        for thread_index in 0..8 {
            let source = Arc::clone(&source);
            scope.spawn(move || {
                for step in 0..100 {
                    let row = thread_index + step * 8;
                    assert_eq!(source.row(row).expect("a valid row")[0], (row * 2) as f64);
                }
            });
        }
    });
}

#[test]
fn eight_threads_read_disjoint_rows_from_an_ipc_file() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrow");
    sk_write_ipc_f64(&path, 800, 50);
    let source = Arc::new(SKArrowIpcMappableSource::<f64>::open(&path).expect("an open source"));
    std::thread::scope(|scope| {
        for thread_index in 0..8 {
            let source = Arc::clone(&source);
            scope.spawn(move || {
                for step in 0..100 {
                    let row = thread_index + step * 8;
                    assert_eq!(source.row(row).expect("a valid row")[0], row as f64);
                }
            });
        }
    });
}
