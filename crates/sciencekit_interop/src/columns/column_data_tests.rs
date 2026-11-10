//! Tests for typed, nullable column storage (spec `dataset-interop-core`).

use super::SKColumnData;
use crate::schema::SKLogicalType;

/// Every variant reports its length and null positions.
#[test]
fn every_variant_reports_length_and_nulls() {
    let columns = vec![
        SKColumnData::Float32(vec![Some(1.0), None, Some(3.0)]),
        SKColumnData::Float64(vec![Some(1.0), None, Some(3.0)]),
        SKColumnData::Integer64(vec![Some(1), None, Some(3)]),
        SKColumnData::Boolean(vec![Some(true), None, Some(false)]),
        SKColumnData::Nominal(vec![
            Some("cat".to_string()),
            None,
            Some("dog".to_string()),
        ]),
        SKColumnData::CategoryCode(vec![Some(0), None, Some(2)]),
    ];

    for column in columns {
        assert_eq!(column.len(), 3);
        assert!(!column.is_empty());
        assert_eq!(column.null_count(), 1, "{}", column.logical_type());
        assert!(column.has_nulls());
        assert_eq!(column.is_null(1), Some(true));
        assert_eq!(column.is_null(0), Some(false));
        assert_eq!(column.is_null(3), None);
    }
}

/// Each variant exposes its logical type.
#[test]
fn variants_expose_their_logical_type() {
    assert_eq!(
        SKColumnData::Float32(Vec::new()).logical_type(),
        SKLogicalType::Float32
    );
    assert_eq!(
        SKColumnData::Float64(Vec::new()).logical_type(),
        SKLogicalType::Float64
    );
    assert_eq!(
        SKColumnData::Integer64(Vec::new()).logical_type(),
        SKLogicalType::Integer64
    );
    assert_eq!(
        SKColumnData::Boolean(Vec::new()).logical_type(),
        SKLogicalType::Boolean
    );
    assert_eq!(
        SKColumnData::Nominal(Vec::new()).logical_type(),
        SKLogicalType::Nominal
    );
    assert_eq!(
        SKColumnData::CategoryCode(Vec::new()).logical_type(),
        SKLogicalType::CategoryCode
    );
}

/// A complete column has no null and an empty one is empty.
#[test]
fn complete_and_empty_columns_are_distinguished() {
    let complete = SKColumnData::Float64(vec![Some(0.5), Some(1.5)]);
    assert_eq!(complete.null_count(), 0);
    assert!(!complete.has_nulls());

    let empty = SKColumnData::Nominal(Vec::new());
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.null_count(), 0);
    assert_eq!(empty.is_null(0), None);
}

/// Cloning a column duplicates the owned values.
#[test]
fn columns_are_owned_values() {
    let column = SKColumnData::Nominal(vec![Some("a".to_string()), None]);
    let clone = column.clone();
    assert_eq!(clone, column);
}
