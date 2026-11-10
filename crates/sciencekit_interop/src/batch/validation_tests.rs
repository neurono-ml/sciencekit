//! Tests for the up-front feature-type validation (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use super::{SKTabularBatch, sk_validate_numeric_feature_columns};
use crate::columns::SKColumnData;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

fn mixed_batch() -> SKTabularBatch {
    let schema = SKTabularSchema::new(vec![
        SKColumnSchema::new("x", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("species", SKLogicalType::Nominal, SKColumnRole::Feature),
        SKColumnSchema::new("code", SKLogicalType::CategoryCode, SKColumnRole::Feature),
    ])
    .unwrap();
    SKTabularBatch::new(
        schema,
        vec![
            SKColumnData::Float64(vec![Some(1.0), None]),
            SKColumnData::Nominal(vec![Some("cat".into()), Some("dog".into())]),
            SKColumnData::CategoryCode(vec![Some(0), None]),
        ],
        0,
        true,
    )
    .unwrap()
}

/// A numeric-only selection passes validation.
#[test]
fn numeric_selection_is_valid() {
    sk_validate_numeric_feature_columns(&mixed_batch(), &[0]).unwrap();
}

/// The first unsupported column is named with a conversion suggestion.
#[test]
fn unsupported_type_is_named_with_a_suggestion() {
    let error = sk_validate_numeric_feature_columns(&mixed_batch(), &[0, 1, 2]).unwrap_err();
    match &error {
        SKError::UnsupportedColumnType {
            column,
            logical_type,
            suggestion,
        } => {
            assert_eq!(column, "species");
            assert_eq!(logical_type, "nominal");
            assert!(suggestion.contains("category"));
        }
        other => panic!("expected an unsupported-column-type error, got {other:?}"),
    }
}

/// The type check runs before element coercion, so an unsupported type wins
/// over a null the policy would otherwise reject.
#[test]
fn type_rejection_precedes_element_processing() {
    let batch = mixed_batch();
    let error = sk_validate_numeric_feature_columns(&batch, &[1, 2]).unwrap_err();
    assert!(matches!(error, SKError::UnsupportedColumnType { .. }));
}

/// An out-of-range selection is reported as an unknown column.
#[test]
fn out_of_range_selection_is_rejected() {
    let error = sk_validate_numeric_feature_columns(&mixed_batch(), &[9]).unwrap_err();
    assert!(matches!(error, SKError::UnknownColumn { .. }));
}
