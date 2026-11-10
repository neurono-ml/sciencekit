//! Tests for explicit feature and target selection (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use super::{SKColumnReference, SKColumnSelection};
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

fn schema() -> SKTabularSchema {
    SKTabularSchema::new(vec![
        SKColumnSchema::new("a", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("b", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("c", SKLogicalType::Float64, SKColumnRole::Ignored),
        SKColumnSchema::new("label", SKLogicalType::Nominal, SKColumnRole::Target),
    ])
    .unwrap()
}

/// Feature order is the declared selection order, not the source order.
#[test]
fn feature_order_is_the_selection_order() {
    let selection = SKColumnSelection::builder()
        .feature("b")
        .feature("a")
        .build()
        .unwrap();
    let resolved = selection.resolve(&schema()).unwrap();
    assert_eq!(resolved.feature_indices(), &[1, 0]);
    assert_eq!(resolved.number_of_features(), 2);
    assert_eq!(resolved.target_index(), None);
    assert!(!selection.has_target());
}

/// Positional selection resolves without consulting names.
#[test]
fn positional_selection_resolves() {
    let selection = SKColumnSelection::builder()
        .feature_at(2)
        .feature_at(0)
        .target_at(3)
        .build()
        .unwrap();
    let resolved = selection.resolve(&schema()).unwrap();
    assert_eq!(resolved.feature_indices(), &[2, 0]);
    assert_eq!(resolved.target_index(), Some(3));
    assert_eq!(
        selection.feature_columns(),
        &[
            SKColumnReference::position(2),
            SKColumnReference::position(0)
        ]
    );
}

/// An unknown name is rejected and named.
#[test]
fn unknown_feature_column_is_rejected() {
    let selection = SKColumnSelection::builder()
        .feature("nope")
        .build()
        .unwrap();
    let error = selection.resolve(&schema()).unwrap_err();
    match &error {
        SKError::UnknownColumn { column, .. } => assert_eq!(column, "nope"),
        other => panic!("expected an unknown-column error, got {other:?}"),
    }
    assert!(error.to_string().contains("nope"));
}

/// An out-of-range position is rejected and named by position.
#[test]
fn out_of_range_position_is_rejected() {
    let selection = SKColumnSelection::builder().feature_at(42).build().unwrap();
    let error = selection.resolve(&schema()).unwrap_err();
    assert!(error.to_string().contains("position 42"));
}

/// Selecting the same column twice is rejected before rows are read.
#[test]
fn duplicate_feature_selection_is_rejected() {
    let selection = SKColumnSelection::builder()
        .feature("a")
        .feature_at(0)
        .build()
        .unwrap();
    let error = selection.resolve(&schema()).unwrap_err();
    match &error {
        SKError::DuplicateColumn { column, .. } => assert_eq!(column, "a"),
        other => panic!("expected a duplicate-column error, got {other:?}"),
    }
}

/// A target also selected as a feature is contradictory and rejected.
#[test]
fn target_also_selected_as_feature_is_rejected() {
    let selection = SKColumnSelection::builder()
        .feature("label")
        .target("label")
        .build()
        .unwrap();
    let error = selection.resolve(&schema()).unwrap_err();
    assert!(matches!(error, SKError::DuplicateColumn { .. }));
}

/// An empty feature selection is rejected at construction.
#[test]
fn empty_feature_selection_is_rejected() {
    let error = SKColumnSelection::builder().build().unwrap_err();
    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "feature_columns"),
        other => panic!("expected an invalid-hyperparameter error, got {other:?}"),
    }
}

/// More than one target is rejected with the unsupported-operation error.
#[test]
fn more_than_one_target_is_rejected() {
    let error = SKColumnSelection::builder()
        .feature("a")
        .target("label")
        .target_at(2)
        .build()
        .unwrap_err();
    match &error {
        SKError::UnsupportedOperation { operation, reason } => {
            assert_eq!(*operation, "column_selection");
            assert!(reason.contains("one target"));
        }
        other => panic!("expected an unsupported-operation error, got {other:?}"),
    }
}

/// A supervised selection exposes its single target.
#[test]
fn supervised_selection_exposes_the_target() {
    let selection = SKColumnSelection::builder()
        .feature("a")
        .target("label")
        .build()
        .unwrap();
    assert!(selection.has_target());
    assert_eq!(
        selection.target_column(),
        Some(&SKColumnReference::name("label"))
    );
}
