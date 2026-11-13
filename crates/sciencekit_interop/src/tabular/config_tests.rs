//! Tests for the tabular source configuration (spec `tabular-sources`).

use sciencekit_common::SKError;

use super::{SKColumnKind, SKTabularSourceConfig};
use crate::batch::SKFeatureLayout;
use crate::missing::SKMissingValuePolicy;

/// The builder preserves declaration order and defaults to dense + reject.
#[test]
fn config_builder_records_declaration_order_and_defaults() {
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .column("label", SKColumnKind::Categorical)
        .column("note", SKColumnKind::Ignored)
        .build()
        .unwrap();

    assert_eq!(
        config
            .columns()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["x", "label", "note"]
    );
    assert_eq!(config.kind_of("label"), Some(SKColumnKind::Categorical));
    assert_eq!(config.kind_of("missing"), None);
    assert_eq!(config.missing_value_policy(), SKMissingValuePolicy::Reject);
    assert_eq!(config.layout(), SKFeatureLayout::Dense);
}

/// An empty configuration is rejected before any row is read.
#[test]
fn config_rejects_an_empty_column_list() {
    let error = SKTabularSourceConfig::builder().build().unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "columns",
            ..
        }
    ));
}

/// A repeated column name is ambiguous, so the builder rejects it.
#[test]
fn config_rejects_a_duplicate_column() {
    let error = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .column("x", SKColumnKind::Categorical)
        .build()
        .unwrap_err();
    match error {
        SKError::DuplicateColumn { column, .. } => assert_eq!(column, "x"),
        other => panic!("expected DuplicateColumn, found {other:?}"),
    }
}

/// The policy and layout builders are carried into the finished config.
#[test]
fn config_carries_policy_and_layout() {
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .missing_value_policy(SKMissingValuePolicy::NumericNaN)
        .layout(SKFeatureLayout::Sparse)
        .build()
        .unwrap();
    assert_eq!(
        config.missing_value_policy(),
        SKMissingValuePolicy::NumericNaN
    );
    assert_eq!(config.layout(), SKFeatureLayout::Sparse);
}

/// Kind classification drives which columns enter which output.
#[test]
fn column_kinds_classify_their_target_output() {
    assert!(SKColumnKind::Numeric.is_numeric());
    assert!(SKColumnKind::Boolean.is_numeric());
    assert!(!SKColumnKind::Categorical.is_numeric());
    assert!(SKColumnKind::Categorical.is_categorical());
    assert!(!SKColumnKind::Ignored.is_numeric());
    assert!(!SKColumnKind::Ignored.is_categorical());
}
