//! Hostile-input suite for the memory-mapped sources (review requirement).
//!
//! Every malformed file must surface as a structured [`SKError`], never a panic
//! and never an out-of-bounds read.

use sciencekit_common::SKError;

use super::descriptor::{SKRawBinaryDescriptor, SKRawBinaryDtype};
use super::numpy_source::SKNumpyMappableSource;
use super::raw_binary_source::SKRawBinaryMappableSource;
use super::test_fixtures::{sk_f64_bytes, sk_write_bytes, sk_write_npy};

#[test]
fn truncated_raw_file_is_a_typed_error() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("truncated.raw");
    sk_write_bytes(&path, &[0_u8; 23]);
    let error = SKRawBinaryMappableSource::<f64>::open(
        &path,
        SKRawBinaryDescriptor::new(3, 2, SKRawBinaryDtype::Float64),
    )
    .expect_err("a truncated raw file is rejected");
    assert!(matches!(error, SKError::MappedLengthMismatch { .. }));
}

#[test]
fn wrong_shape_descriptor_is_a_typed_error() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("shape.raw");
    sk_write_bytes(&path, &sk_f64_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let error = SKRawBinaryMappableSource::<f64>::open(
        &path,
        // The file carries a 2x2 matrix; the descriptor claims 1x8.
        SKRawBinaryDescriptor::new(1, 8, SKRawBinaryDtype::Float64),
    )
    .expect_err("a wrong shape descriptor is rejected");
    match error {
        SKError::MappedLengthMismatch {
            path: named,
            expected_bytes,
            actual_bytes,
        } => {
            assert_eq!(named, path.display().to_string());
            assert_eq!(expected_bytes, 64);
            assert_eq!(actual_bytes, 32);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn absurd_npy_dimension_is_a_typed_error() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("absurd.npy");
    // Two axes whose product fits in `usize` but whose byte length does not.
    let huge_rows = 1_usize << 61;
    sk_write_npy(
        &path,
        "<f8",
        huge_rows,
        2,
        false,
        true,
        &sk_f64_bytes(&[1.0]),
    );
    let error = SKNumpyMappableSource::<f64>::open(&path)
        .expect_err("an absurd dimension is rejected, not multiplied unchecked");
    assert!(matches!(
        error,
        SKError::MalformedMappedHeader { .. } | SKError::MappedLengthMismatch { .. }
    ));
}
