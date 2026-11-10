//! Tests for the interop error variants of the central taxonomy (spec
//! `dataset-interop-core`).

use super::SKError;

/// An unknown tabular column names the column and a way to fix the selection.
#[test]
fn unknown_column_names_the_column_and_a_suggestion() {
    let err = SKError::UnknownColumn {
        column: "age".to_string(),
        suggestion: "check the schema for the exact column name".to_string(),
    };
    let message = format!("{err}");
    assert!(message.contains("age"));
    assert!(message.contains("check the schema"));
}

/// A duplicate tabular column names the repeated column.
#[test]
fn duplicate_column_names_the_column() {
    let err = SKError::DuplicateColumn {
        column: "species".to_string(),
        suggestion: "remove the repeated feature reference".to_string(),
    };
    assert!(format!("{err}").contains("species"));
}

/// An unsupported column type names the column, the type and the suggestion.
#[test]
fn unsupported_column_type_names_column_type_and_suggestion() {
    let err = SKError::UnsupportedColumnType {
        column: "species".to_string(),
        logical_type: "nominal".to_string(),
        suggestion: "encode it as category codes".to_string(),
    };
    let message = format!("{err}");
    assert!(message.contains("species"));
    assert!(message.contains("nominal"));
    assert!(message.contains("category codes"));
}

/// A null under the reject policy names the column and the policy.
#[test]
fn null_value_names_the_column_and_the_policy() {
    let err = SKError::NullValueInColumn {
        column: "price".to_string(),
        policy: "reject",
    };
    let message = format!("{err}");
    assert!(message.contains("price"));
    assert!(message.contains("reject"));
}

/// Uneven column lengths name the offender and both lengths.
#[test]
fn uneven_column_lengths_name_the_column_and_lengths() {
    let err = SKError::UnevenColumnLengths {
        column: "y".to_string(),
        expected: 3,
        found: 2,
    };
    let message = format!("{err}");
    assert!(message.contains('y'));
    assert!(message.contains('3'));
    assert!(message.contains('2'));
}

/// A schema mismatch reports what was expected and what arrived.
#[test]
fn schema_mismatch_reports_both_sides() {
    let err = SKError::SchemaMismatch {
        expected: "3 columns".to_string(),
        found: "2 columns".to_string(),
    };
    let message = format!("{err}");
    assert!(message.contains("expected 3 columns"));
    assert!(message.contains("found 2 columns"));
}
