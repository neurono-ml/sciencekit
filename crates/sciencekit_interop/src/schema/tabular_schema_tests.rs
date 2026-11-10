//! Tests for the ordered tabular schema (spec `dataset-interop-core`).

use sciencekit_common::SKError;

use super::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

fn column(name: &str, logical_type: SKLogicalType, role: SKColumnRole) -> SKColumnSchema {
    SKColumnSchema::new(name, logical_type, role)
}

fn example_schema() -> SKTabularSchema {
    SKTabularSchema::new(vec![
        column("b", SKLogicalType::Float64, SKColumnRole::Feature),
        column("a", SKLogicalType::Float64, SKColumnRole::Feature),
        column("label", SKLogicalType::Nominal, SKColumnRole::Target),
        column("ignored_id", SKLogicalType::Integer64, SKColumnRole::Ignored),
    ])
    .unwrap()
}

/// Lookup is deterministic and preserves the declared source order.
#[test]
fn lookup_is_deterministic_and_ordered() {
    let schema = example_schema();
    assert_eq!(schema.len(), 4);
    assert_eq!(schema.column_index("b"), Some(0));
    assert_eq!(schema.column_index("a"), Some(1));
    assert_eq!(schema.column_index("label"), Some(2));
    assert_eq!(schema.column_index("missing"), None);
    assert_eq!(schema.column("a").unwrap().name(), "a");
    assert_eq!(schema.column_at(2).unwrap().name(), "label");
    assert_eq!(schema.column_at(9), None);

    let names: Vec<&str> = schema.columns().iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["b", "a", "label", "ignored_id"]);
}

/// Duplicate names are rejected before any row can be read.
#[test]
fn duplicate_column_names_are_rejected() {
    let error = SKTabularSchema::new(vec![
        column("x", SKLogicalType::Float64, SKColumnRole::Feature),
        column("x", SKLogicalType::Float64, SKColumnRole::Feature),
    ])
    .unwrap_err();
    match &error {
        SKError::DuplicateColumn { column, suggestion } => {
            assert_eq!(column, "x");
            assert!(!suggestion.is_empty());
        }
        other => panic!("expected a duplicate-column error, got {other:?}"),
    }
    assert!(error.to_string().contains('x'));
}

/// An empty schema is rejected — a table always names at least one column.
#[test]
fn empty_schema_is_rejected() {
    let error = SKTabularSchema::new(Vec::new()).unwrap_err();
    assert!(matches!(error, SKError::SchemaMismatch { .. }));
}

/// Roles partition the declared columns deterministically.
#[test]
fn roles_partition_the_columns() {
    let schema = example_schema();
    assert_eq!(schema.column_indices_with_role(SKColumnRole::Feature), vec![0, 1]);
    assert_eq!(schema.column_indices_with_role(SKColumnRole::Target), vec![2]);
    assert_eq!(
        schema.column_indices_with_role(SKColumnRole::Ignored),
        vec![3]
    );
}

/// A schema is a value: cloning and comparing preserves the declaration.
#[test]
fn schema_is_a_comparable_value() {
    let schema = example_schema();
    assert_eq!(schema.clone(), schema);
    assert!(!schema.is_empty());
}
