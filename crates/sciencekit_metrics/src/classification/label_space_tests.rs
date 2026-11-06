//! TDD tests for the classification label-space resolution
//! (spec `evaluation-metrics`).

use sciencekit_common::SKError;

use super::SKLabelSpace;

/// An explicit label list fixes the class order independently of observation order.
#[test]
fn explicit_labels_fix_the_class_order() {
    let space = SKLabelSpace::from_explicit(&[2, 0, 1]).expect("valid label list");
    assert_eq!(space.labels(), &[0, 1, 2]);
}

/// Without an explicit list, the space is the sorted union of observed labels.
#[test]
fn default_label_space_is_the_observed_union() {
    let space = SKLabelSpace::from_observed(&[0, 0, 2], &[1, 2, 0]);
    assert_eq!(space.labels(), &[0, 1, 2]);
}

/// Duplicated explicit labels are rejected as an invalid hyperparameter.
#[test]
fn explicit_labels_reject_duplicates() {
    let error = SKLabelSpace::from_explicit(&[1, 1, 2]).expect_err("duplicates must fail");
    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "labels"),
        other => panic!("expected invalid-hyperparameter, got {other:?}"),
    }
}

/// An empty explicit label list is rejected.
#[test]
fn explicit_labels_reject_an_empty_list() {
    let error = SKLabelSpace::from_explicit(&[]).expect_err("an empty list must fail");
    assert!(matches!(error, SKError::InvalidHyperparameter { .. }));
}

/// A label outside the explicit space reports its value.
#[test]
fn unknown_label_carries_the_offending_value() {
    let space = SKLabelSpace::from_explicit(&[0, 1]).expect("valid label list");
    let error = space.index_of(7).expect_err("label 7 is outside the space");
    match error {
        SKError::UnknownLabel { label } => assert_eq!(label, 7),
        other => panic!("expected unknown-label, got {other:?}"),
    }
}

/// Every observed label has an index inside the resolved space.
#[test]
fn observed_labels_resolve_to_indices() {
    let space = SKLabelSpace::from_observed(&[5, 3], &[5]);
    assert_eq!(space.index_of(3).expect("observed"), 0);
    assert_eq!(space.index_of(5).expect("observed"), 1);
    assert_eq!(space.len(), 2);
}
