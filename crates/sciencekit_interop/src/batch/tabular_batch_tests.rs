//! Tests for the column-major typed batch (spec `dataset-interop-core`).

use sciencekit_common::SKError;

use super::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

pub(crate) fn schema() -> SKTabularSchema {
    SKTabularSchema::new(vec![
        SKColumnSchema::new("x", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("y", SKLogicalType::Integer64, SKColumnRole::Feature),
        SKColumnSchema::new("label", SKLogicalType::Nominal, SKColumnRole::Target),
    ])
    .unwrap()
}

fn batch() -> SKTabularBatch {
    SKTabularBatch::new(
        schema(),
        vec![
            SKColumnData::Float64(vec![Some(1.0), Some(2.0)]),
            SKColumnData::Integer64(vec![Some(10), None]),
            SKColumnData::Nominal(vec![Some("a".into()), Some("b".into())]),
        ],
        3,
        true,
    )
    .unwrap()
}

/// A valid batch reports its shape, metadata and named columns.
#[test]
fn batch_reports_shape_and_metadata() {
    let batch = batch();
    assert_eq!(batch.number_of_rows(), 2);
    assert_eq!(batch.number_of_columns(), 3);
    assert!(!batch.is_empty());
    assert_eq!(batch.position(), 3);
    assert!(batch.is_final());
    assert_eq!(batch.column(0), Some(&SKColumnData::Float64(vec![Some(1.0), Some(2.0)])));
    assert_eq!(
        batch.column_named("y").unwrap().logical_type(),
        SKLogicalType::Integer64
    );
    assert!(batch.column_named("missing").is_none());
    assert!(batch.column(9).is_none());
}

/// Columns of different lengths are rejected and the offender is named.
#[test]
fn uneven_column_lengths_are_rejected() {
    let error = SKTabularBatch::new(
        schema(),
        vec![
            SKColumnData::Float64(vec![Some(1.0), Some(2.0)]),
            SKColumnData::Integer64(vec![Some(10)]),
            SKColumnData::Nominal(vec![Some("a".into()), Some("b".into())]),
        ],
        0,
        true,
    )
    .unwrap_err();
    match &error {
        SKError::UnevenColumnLengths {
            column,
            expected,
            found,
        } => {
            assert_eq!(column, "y");
            assert_eq!(*expected, 2);
            assert_eq!(*found, 1);
        }
        other => panic!("expected an uneven-length error, got {other:?}"),
    }
    assert!(error.to_string().contains('y'));
}

/// A column count different from the schema is a schema mismatch.
#[test]
fn column_count_mismatch_is_rejected() {
    let error = SKTabularBatch::new(
        schema(),
        vec![SKColumnData::Float64(vec![Some(1.0)])],
        0,
        true,
    )
    .unwrap_err();
    assert!(matches!(error, SKError::SchemaMismatch { .. }));
}

/// A column whose data type disagrees with the schema is rejected.
#[test]
fn logical_type_mismatch_is_rejected() {
    let error = SKTabularBatch::new(
        schema(),
        vec![
            SKColumnData::Integer64(vec![Some(1)]),
            SKColumnData::Integer64(vec![Some(10)]),
            SKColumnData::Nominal(vec![Some("a".into())]),
        ],
        0,
        true,
    )
    .unwrap_err();
    match &error {
        SKError::SchemaMismatch { expected, .. } => assert!(expected.contains('x')),
        other => panic!("expected a schema mismatch, got {other:?}"),
    }
}

/// An empty batch is valid: zero rows, full schema, no null handling yet.
#[test]
fn empty_batch_is_valid() {
    let batch = SKTabularBatch::new(
        schema(),
        vec![
            SKColumnData::Float64(Vec::new()),
            SKColumnData::Integer64(Vec::new()),
            SKColumnData::Nominal(Vec::new()),
        ],
        0,
        true,
    )
    .unwrap();
    assert_eq!(batch.number_of_rows(), 0);
    assert!(batch.is_empty());
}

/// Projection preserves the requested order and the streaming metadata.
#[test]
fn projection_preserves_order_and_metadata() {
    let projected = batch().project(&[2, 0]).unwrap();
    assert_eq!(projected.number_of_columns(), 2);
    assert_eq!(projected.schema().columns()[0].name(), "label");
    assert_eq!(projected.schema().columns()[1].name(), "x");
    assert_eq!(projected.position(), 3);
    assert!(projected.is_final());
}

/// Projection rejects an out-of-range column and a repeated one.
#[test]
fn projection_rejects_invalid_positions() {
    let error = batch().project(&[0, 7]).unwrap_err();
    assert!(matches!(error, SKError::UnknownColumn { .. }));
    let duplicate = batch().project(&[0, 0]).unwrap_err();
    assert!(matches!(duplicate, SKError::DuplicateColumn { .. }));
}
