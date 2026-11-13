//! TDD suite for the NumPy `.npy` memory-mapped source (tasks 2.1–2.5, 4.3).

use sciencekit_common::{SKDataView, SKError, SKLazySource, SKMappableSource};

use super::numpy_source::SKNumpyMappableSource;
use super::test_fixtures::{sk_f64_bytes, sk_write_bytes, sk_write_npy};

fn write_f64(path: &std::path::Path, rows: usize, columns: usize) {
    let values: Vec<f64> = (0..rows * columns).map(|index| index as f64).collect();
    sk_write_npy(
        path,
        "<f8",
        rows,
        columns,
        false,
        true,
        &sk_f64_bytes(&values),
    );
}

#[test]
fn float64_rows_alias_the_mapping() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.npy");
    write_f64(&path, 3, 2);
    let source = SKNumpyMappableSource::<f64>::open(&path).expect("an open .npy source");
    assert!(source.is_zero_copy());
    assert_eq!(source.number_of_rows(), 3);
    let row = source.row(1).unwrap();
    assert_eq!(row.to_vec(), vec![2.0, 3.0]);
    assert_eq!(
        row.as_ptr() as usize,
        source.mapped_payload_address().unwrap() + 2 * 8
    );
}

#[test]
fn float32_rows_alias_the_mapping() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.npy");
    let values: Vec<f32> = (0..6).map(|index| index as f32 + 0.25).collect();
    sk_write_npy(
        &path,
        "<f4",
        2,
        3,
        false,
        true,
        &super::test_fixtures::sk_f32_bytes(&values),
    );
    let source = SKNumpyMappableSource::<f32>::open(&path).expect("an open .npy source");
    assert_eq!(source.row(1).unwrap().to_vec(), vec![3.25, 4.25, 5.25]);
}

#[test]
fn malformed_header_is_rejected_without_panicking() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("corrupt.npy");
    sk_write_bytes(&path, b"not a numpy file at all");
    let error = SKNumpyMappableSource::<f64>::open(&path).expect_err("a corrupt header");
    match error {
        SKError::MalformedMappedHeader {
            path: named,
            format,
            ..
        } => {
            assert_eq!(named, path.display().to_string());
            assert_eq!(format, "npy");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn truncated_header_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("truncated.npy");
    sk_write_bytes(&path, b"\x93NUMPY\x01\x00\xff\x7f");
    let error = SKNumpyMappableSource::<f64>::open(&path).expect_err("a truncated header");
    assert!(matches!(error, SKError::MalformedMappedHeader { .. }));
}

#[test]
fn fortran_order_is_rejected_with_a_c_order_suggestion() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("fortran.npy");
    sk_write_npy(&path, "<f8", 2, 3, true, true, &sk_f64_bytes(&[0.0; 6]));
    let error = SKNumpyMappableSource::<f64>::open(&path).expect_err("Fortran order is rejected");
    match error {
        SKError::UnsupportedOperation { reason, .. } => {
            assert!(reason.contains("ascontiguousarray"))
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn int64_without_cast_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("integers.npy");
    sk_write_npy(
        &path,
        "<i8",
        2,
        2,
        false,
        true,
        &super::test_fixtures::sk_i64_bytes(&[1, 2, 3, 4]),
    );
    let error = SKNumpyMappableSource::<f64>::open(&path).expect_err("i64 needs an explicit cast");
    match error {
        SKError::UnsupportedMappedDtype { dtype, .. } => assert_eq!(dtype, "i8"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn int64_with_cast_materializes_values() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("integers.npy");
    sk_write_npy(
        &path,
        "<i8",
        2,
        2,
        false,
        true,
        &super::test_fixtures::sk_i64_bytes(&[1, 2, 3, 4]),
    );
    let source =
        SKNumpyMappableSource::<f64>::open_with_cast(&path).expect("the casting path materializes");
    assert!(!source.is_zero_copy());
    assert_eq!(source.row(1).unwrap().to_vec(), vec![3.0, 4.0]);
}

#[test]
fn misaligned_header_payload_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("misaligned.npy");
    sk_write_npy(
        &path,
        "<f8",
        2,
        2,
        false,
        false,
        &sk_f64_bytes(&[1.0, 2.0, 3.0, 4.0]),
    );
    let error =
        SKNumpyMappableSource::<f64>::open(&path).expect_err("an unaligned payload is rejected");
    assert!(matches!(error, SKError::MisalignedMappedData { .. }));
}

#[test]
fn length_mismatch_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("short.npy");
    sk_write_npy(&path, "<f8", 2, 2, false, true, &sk_f64_bytes(&[1.0, 2.0]));
    let error = SKNumpyMappableSource::<f64>::open(&path).expect_err("a short payload");
    assert!(matches!(error, SKError::MappedLengthMismatch { .. }));
}

#[test]
fn big_endian_npy_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("big.npy");
    sk_write_npy(&path, ">f8", 1, 2, false, true, &sk_f64_bytes(&[0.0, 0.0]));
    let error = SKNumpyMappableSource::<f64>::open(&path).expect_err("big-endian data");
    assert!(matches!(error, SKError::NonLittleEndianMappedData { .. }));
}

#[test]
fn sequential_blocks_match_random_access() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.npy");
    write_f64(&path, 10_000, 2);
    let mut sequential = SKNumpyMappableSource::<f64>::open(&path)
        .expect("an open source")
        .with_block_size(512)
        .expect("a non-zero block size");
    let mut seen = Vec::new();
    let mut finals = 0;
    while let Some(batch) = sequential.next_batch().unwrap() {
        assert!(batch.number_of_rows() <= 512);
        if batch.is_final() {
            finals += 1;
        }
        match batch.view() {
            SKDataView::Dense(view) => seen.extend(view.iter().copied()),
            _ => panic!("expected a dense block"),
        }
    }
    assert_eq!(finals, 1);
    assert_eq!(seen.len(), 20_000);
    let random = SKNumpyMappableSource::<f64>::open(&path).expect("a second open");
    assert_eq!(seen[0], random.row(0).unwrap()[0]);
    assert_eq!(seen[19_999], random.row(9_999).unwrap()[1]);
}
