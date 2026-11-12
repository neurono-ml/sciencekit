//! TDD suite for the raw binary descriptor (tasks 1.2).

use super::descriptor::{SKRawBinaryDescriptor, SKRawBinaryDtype};
use sciencekit_common::SKError;

#[test]
fn explicit_descriptor_reports_shape_and_dtype() {
    let descriptor = SKRawBinaryDescriptor::new(4, 3, SKRawBinaryDtype::Float64);
    assert_eq!(descriptor.number_of_rows(), 4);
    assert_eq!(descriptor.number_of_columns(), 3);
    assert_eq!(descriptor.dtype(), SKRawBinaryDtype::Float64);
    assert!(descriptor.is_little_endian());
    assert_eq!(descriptor.required_byte_length().unwrap(), 4 * 3 * 8);
}

#[test]
fn descriptor_round_trips_through_json() {
    let descriptor = SKRawBinaryDescriptor::new(2, 5, SKRawBinaryDtype::Float32);
    let parsed = SKRawBinaryDescriptor::from_json(&descriptor.to_json(), "inline")
        .expect("a valid descriptor");
    assert_eq!(parsed, descriptor);
}

#[test]
fn descriptor_reads_the_documented_json_shape() {
    let parsed = SKRawBinaryDescriptor::from_json(
        r#"{"rows": 10, "columns": 2, "dtype": "f64"}"#,
        "sidecar.json",
    )
    .expect("the documented descriptor shape");
    assert_eq!(
        parsed,
        SKRawBinaryDescriptor::new(10, 2, SKRawBinaryDtype::Float64)
    );
}

#[test]
fn descriptor_reads_a_sidecar_file() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.json");
    std::fs::write(&path, r#"{"rows": 1, "columns": 1, "dtype": "f32"}"#).unwrap();
    let descriptor = SKRawBinaryDescriptor::from_sidecar(&path).expect("a sidecar descriptor");
    assert_eq!(descriptor.dtype(), SKRawBinaryDtype::Float32);
}

#[test]
fn invalid_descriptor_is_rejected_with_the_source_name() {
    let error = SKRawBinaryDescriptor::from_json(r#"{"rows": 10}"#, "matrix.json")
        .expect_err("an incomplete descriptor is rejected");
    match error {
        SKError::MalformedMappedHeader { path, format, .. } => {
            assert_eq!(path, "matrix.json");
            assert_eq!(format, "raw-binary-descriptor");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn big_endian_descriptor_is_rejected_at_open() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.raw");
    super::test_fixtures::sk_write_bytes(&path, &[0_u8; 16]);
    let descriptor = SKRawBinaryDescriptor {
        rows: 2,
        columns: 1,
        dtype: SKRawBinaryDtype::Float64,
        endianness: super::SKRawBinaryEndianness::Big,
    };
    let error = super::SKRawBinaryMappableSource::<f64>::open(&path, descriptor)
        .expect_err("big-endian data is rejected");
    assert!(matches!(error, SKError::NonLittleEndianMappedData { .. }));
}
