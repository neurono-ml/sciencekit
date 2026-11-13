//! TDD suite for the raw binary memory-mapped source (tasks 1.3–1.5, 4.2).

use std::path::Path;

use sciencekit_common::{SKDataView, SKError, SKLazySource, SKMappableSource};

use super::descriptor::{SKRawBinaryDescriptor, SKRawBinaryDtype};
use super::raw_binary_source::SKRawBinaryMappableSource;
use super::test_fixtures::{sk_f32_bytes, sk_f64_bytes, sk_write_bytes};

fn descriptor(rows: usize, columns: usize, dtype: SKRawBinaryDtype) -> SKRawBinaryDescriptor {
    SKRawBinaryDescriptor::new(rows, columns, dtype)
}

fn write_f64_matrix(path: &Path, rows: usize, columns: usize) {
    let values: Vec<f64> = (0..rows * columns).map(|index| index as f64).collect();
    sk_write_bytes(path, &sk_f64_bytes(&values));
}

#[test]
fn random_access_returns_the_requested_row() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.raw");
    write_f64_matrix(&path, 4, 3);
    let source =
        SKRawBinaryMappableSource::<f64>::open(&path, descriptor(4, 3, SKRawBinaryDtype::Float64))
            .expect("an open source");
    assert_eq!(source.number_of_rows(), 4);
    assert_eq!(source.row(0).unwrap().to_vec(), vec![0.0, 1.0, 2.0]);
    assert_eq!(source.row(2).unwrap().to_vec(), vec![6.0, 7.0, 8.0]);
    assert_eq!(source.row(3).unwrap().to_vec(), vec![9.0, 10.0, 11.0]);
}

#[test]
fn row_access_does_not_scan_or_copy() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.raw");
    write_f64_matrix(&path, 1_000_000, 2);
    let source = SKRawBinaryMappableSource::<f64>::open(
        &path,
        descriptor(1_000_000, 2, SKRawBinaryDtype::Float64),
    )
    .expect("an open source");
    let row = source.row(900_000).unwrap();
    assert_eq!(row.to_vec(), vec![1_800_000.0, 1_800_001.0]);
    assert_eq!(
        row.as_ptr() as usize,
        source.mapped_payload_address() + 900_000 * 2 * 8
    );
}

#[test]
fn out_of_bounds_row_is_a_structured_error() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.raw");
    write_f64_matrix(&path, 2, 2);
    let source =
        SKRawBinaryMappableSource::<f64>::open(&path, descriptor(2, 2, SKRawBinaryDtype::Float64))
            .expect("an open source");
    assert!(source.row(2).is_err());
}

#[test]
fn f32_rows_are_viewed_without_copying() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.raw");
    sk_write_bytes(&path, &sk_f32_bytes(&[1.5, 2.5, 3.5, 4.5]));
    let source =
        SKRawBinaryMappableSource::<f32>::open(&path, descriptor(2, 2, SKRawBinaryDtype::Float32))
            .expect("an open f32 source");
    assert_eq!(source.row(1).unwrap().to_vec(), vec![3.5, 4.5]);
}

#[test]
fn shorter_file_is_rejected_naming_the_file() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("short.raw");
    sk_write_bytes(&path, &[0_u8; 15]);
    let error =
        SKRawBinaryMappableSource::<f64>::open(&path, descriptor(2, 1, SKRawBinaryDtype::Float64))
            .expect_err("a short file is rejected");
    match error {
        SKError::MappedLengthMismatch {
            path: named,
            expected_bytes,
            actual_bytes,
        } => {
            assert_eq!(named, path.display().to_string());
            assert_eq!(expected_bytes, 16);
            assert_eq!(actual_bytes, 15);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn longer_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("long.raw");
    write_f64_matrix(&path, 3, 1);
    let error =
        SKRawBinaryMappableSource::<f64>::open(&path, descriptor(2, 1, SKRawBinaryDtype::Float64))
            .expect_err("a long file is rejected");
    assert!(matches!(error, SKError::MappedLengthMismatch { .. }));
}

#[test]
fn dtype_mismatch_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("int.raw");
    sk_write_bytes(&path, &[0_u8; 16]);
    let error =
        SKRawBinaryMappableSource::<f64>::open(&path, descriptor(2, 1, SKRawBinaryDtype::Int64))
            .expect_err("an i64 file cannot back an f64 model");
    match error {
        SKError::UnsupportedMappedDtype {
            path: named, dtype, ..
        } => {
            assert_eq!(named, path.display().to_string());
            assert_eq!(dtype, "i64");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn missing_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("absent.raw");
    let error =
        SKRawBinaryMappableSource::<f64>::open(&path, descriptor(1, 1, SKRawBinaryDtype::Float64))
            .expect_err("a missing file is rejected");
    assert!(matches!(error, SKError::MissingMappedFile { .. }));
}

#[test]
fn misaligned_buffer_is_rejected_by_the_view_helper() {
    let path = Path::new("unaligned.bin");
    let bytes = [0_u8; 32];
    let misaligned = &bytes[1..17];
    let error = super::alignment::sk_check_alignment::<f64>(misaligned, path)
        .expect_err("an odd address cannot back an f64 view");
    assert!(matches!(error, SKError::MisalignedMappedData { .. }));
}

#[test]
fn sequential_blocks_respect_the_block_size_and_mark_the_last() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.raw");
    write_f64_matrix(&path, 10, 2);
    let mut source =
        SKRawBinaryMappableSource::<f64>::open(&path, descriptor(10, 2, SKRawBinaryDtype::Float64))
            .expect("an open source")
            .with_block_size(4)
            .expect("a non-zero block size");
    let mut seen = Vec::new();
    let mut finals = 0;
    while let Some(batch) = source.next_batch().unwrap() {
        assert!(batch.number_of_rows() <= 4);
        if batch.is_final() {
            finals += 1;
        }
        match batch.view() {
            SKDataView::Dense(view) => seen.extend(view.iter().copied()),
            _ => panic!("expected a dense block"),
        }
    }
    assert_eq!(finals, 1);
    assert_eq!(seen.len(), 20);
    assert_eq!(seen[0], 0.0);
    assert_eq!(seen[19], 19.0);
}

#[test]
fn a_large_file_streams_in_bounded_batches() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("large.raw");
    let rows = 10_000;
    let columns = 8;
    write_f64_matrix(&path, rows, columns);
    let mut source = SKRawBinaryMappableSource::<f64>::open(
        &path,
        descriptor(rows, columns, SKRawBinaryDtype::Float64),
    )
    .expect("an open source")
    .with_block_size(512)
    .expect("a non-zero block size");
    let budget = 512 * columns * std::mem::size_of::<f64>();
    let mut batches = 0;
    while let Some(batch) = source.next_batch().unwrap() {
        assert!(batch.byte_size() <= budget);
        assert!(batch.number_of_rows() <= 512);
        batches += 1;
    }
    assert_eq!(batches, 20);
}
