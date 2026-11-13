//! Tests for the neutral column schema (spec `dataset-interop-core`).

use std::str::FromStr;

use sciencekit_common::SKError;

use super::{SKColumnRole, SKColumnSchema, SKLogicalType};

/// Every role parses from its stable name and renders back to it.
#[test]
fn role_names_round_trip() {
    for role in [
        SKColumnRole::Feature,
        SKColumnRole::Target,
        SKColumnRole::Ignored,
    ] {
        assert_eq!(SKColumnRole::from_str(role.name()).unwrap(), role);
        assert_eq!(role.to_string(), role.name());
    }
}

/// An unknown role name is rejected, never defaulted.
#[test]
fn unknown_role_is_rejected() {
    let error = SKColumnRole::from_str("response").unwrap_err();
    assert!(
        error.to_string().contains("response"),
        "the error must name the rejected role: {error}"
    );
    match error {
        SKError::Conversion(_) => {}
        other => panic!("expected a conversion error, got {other:?}"),
    }
}

/// Every logical type renders its stable name and parses back.
#[test]
fn logical_type_names_round_trip() {
    for logical_type in [
        SKLogicalType::Float32,
        SKLogicalType::Float64,
        SKLogicalType::Integer64,
        SKLogicalType::Boolean,
        SKLogicalType::Nominal,
        SKLogicalType::CategoryCode,
    ] {
        assert_eq!(
            SKLogicalType::from_name(logical_type.name()).unwrap(),
            logical_type
        );
        assert_eq!(logical_type.to_string(), logical_type.name());
    }
}

/// An unknown logical type name is rejected with the accepted vocabulary.
#[test]
fn unknown_logical_type_is_rejected() {
    let error = SKLogicalType::from_name("temporal").unwrap_err();
    let message = error.to_string();
    assert!(message.contains("temporal"), "the type must be named");
    assert!(
        message.contains("float64") && message.contains("category_code"),
        "the accepted vocabulary must be listed: {message}"
    );
}

/// Numeric and categorical classification follows the logical type.
#[test]
fn logical_types_classify_as_numeric_or_categorical() {
    assert!(SKLogicalType::Float32.is_numeric());
    assert!(SKLogicalType::Float64.is_numeric());
    assert!(SKLogicalType::Integer64.is_numeric());
    assert!(SKLogicalType::Boolean.is_numeric());
    assert!(!SKLogicalType::Nominal.is_numeric());
    assert!(SKLogicalType::Nominal.is_categorical());
    assert!(SKLogicalType::CategoryCode.is_categorical());
    assert!(!SKLogicalType::Float64.is_categorical());
}

/// A schema column exposes exactly what was declared and no data.
#[test]
fn column_schema_exposes_name_type_and_role() {
    let column = SKColumnSchema::new("age", SKLogicalType::Float64, SKColumnRole::Feature);
    assert_eq!(column.name(), "age");
    assert_eq!(column.logical_type(), SKLogicalType::Float64);
    assert_eq!(column.role(), SKColumnRole::Feature);
    assert!(column.is_numeric());
    assert!(!column.is_categorical());

    let label = SKColumnSchema::new("species", SKLogicalType::Nominal, SKColumnRole::Target);
    assert_eq!(label.name(), "species");
    assert!(label.is_categorical());
    assert!(!label.is_numeric());
}

/// A schema column is cloneable and comparable, so schemas can be copied.
#[test]
fn column_schema_is_a_value() {
    let column = SKColumnSchema::new("x", SKLogicalType::Integer64, SKColumnRole::Ignored);
    assert_eq!(column.clone(), column);
}
