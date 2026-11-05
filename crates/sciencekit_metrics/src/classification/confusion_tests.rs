//! TDD tests for the confusion matrix (spec `evaluation-metrics`).

use sciencekit_common::SKError;

use super::sk_confusion_matrix;

/// Counts match the observations, rows true and columns predicted.
#[test]
fn counts_match_the_observations() {
    let matrix = sk_confusion_matrix(&[0, 0, 0], &[0, 1, 0], None).expect("valid input");
    assert_eq!(matrix.labels(), &[0, 1]);
    assert_eq!(matrix.shape(), (2, 2));
    assert_eq!(matrix.get(0, 0), Some(2));
    assert_eq!(matrix.get(0, 1), Some(1));
    assert_eq!(matrix.get(1, 0), Some(0));
    assert_eq!(matrix.get(1, 1), Some(0));
    assert_eq!(matrix.number_of_observations(), 3);
}

/// A resolved class absent from the predictions keeps a zero column.
#[test]
fn resolved_class_absent_from_predictions_has_a_zero_column() {
    let matrix = sk_confusion_matrix(&[0, 0], &[0, 0], Some(&[0, 1])).expect("valid input");
    assert_eq!(matrix.labels(), &[0, 1]);
    assert_eq!(matrix.shape(), (2, 2));
    assert_eq!(matrix.get(0, 1), Some(0));
    assert_eq!(matrix.get(1, 1), Some(0));
}

/// An explicit label list fixes the class order.
#[test]
fn explicit_labels_fix_the_class_order() {
    let matrix = sk_confusion_matrix(&[2, 0], &[0, 2], Some(&[2, 0, 1])).expect("valid input");
    assert_eq!(matrix.labels(), &[0, 1, 2]);
    assert_eq!(matrix.get(2, 0), Some(1));
    assert_eq!(matrix.get(0, 2), Some(1));
}

/// A label outside the explicit space is rejected.
#[test]
fn unknown_labels_are_rejected() {
    let error = sk_confusion_matrix(&[2], &[2], Some(&[0, 1])).expect_err("label 2 is unknown");
    assert!(matches!(error, SKError::UnknownLabel { label: 2 }));
}

/// Different lengths are rejected as a shape error.
#[test]
fn length_mismatch_is_rejected() {
    let error = sk_confusion_matrix(&[0, 1], &[0], None).expect_err("lengths differ");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Empty observations are rejected.
#[test]
fn empty_observations_are_rejected() {
    let error = sk_confusion_matrix(&[], &[], None).expect_err("no observations");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}
