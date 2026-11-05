//! TDD tests for the classification streaming accumulator (spec `evaluation-metrics`).

use sciencekit_common::SKError;

use super::SKClassificationAccumulator;
use crate::classification::sk_confusion_matrix;

/// Batched accumulation equals a whole-input confusion matrix.
#[test]
fn batched_accumulation_equals_whole_input() {
    let true_labels = [0_i64, 0, 0, 1, 1, 2];
    let predicted_labels = [0_i64, 1, 0, 1, 0, 2];
    let mut accumulator = SKClassificationAccumulator::new();
    for (true_label, predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        accumulator.observe(*true_label, *predicted_label);
    }
    let matrix = accumulator
        .confusion_matrix(None)
        .expect("non-empty accumulator");
    let expected = sk_confusion_matrix(&true_labels, &predicted_labels, None).expect("valid input");
    assert_eq!(matrix.labels(), expected.labels());
    assert_eq!(matrix.counts(), expected.counts());
    assert_eq!(accumulator.number_of_observations(), 6);
}

/// Merging two accumulators equals one pass over the concatenation.
#[test]
fn merging_two_accumulators_equals_concatenation() {
    let true_labels = [0_i64, 1, 1, 2, 2, 0];
    let predicted_labels = [0_i64, 1, 0, 2, 1, 1];
    let mut left = SKClassificationAccumulator::new();
    for (true_label, predicted_label) in true_labels[..3].iter().zip(predicted_labels[..3].iter()) {
        left.observe(*true_label, *predicted_label);
    }
    let mut right = SKClassificationAccumulator::new();
    for (true_label, predicted_label) in true_labels[3..].iter().zip(predicted_labels[3..].iter()) {
        right.observe(*true_label, *predicted_label);
    }
    left.merge(&right);
    let merged = left.confusion_matrix(None).expect("non-empty accumulator");
    let expected = sk_confusion_matrix(&true_labels, &predicted_labels, None).expect("valid input");
    assert_eq!(merged.labels(), expected.labels());
    assert_eq!(merged.counts(), expected.counts());
    assert_eq!(left.number_of_observations(), 6);
}

/// An empty accumulator is rejected on finalization.
#[test]
fn empty_accumulator_is_rejected() {
    let accumulator = SKClassificationAccumulator::new();
    let error = accumulator
        .confusion_matrix(None)
        .expect_err("an empty accumulator has no observations");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// Unseen labels extend the class space in ascending order.
#[test]
fn unseen_labels_extend_the_class_space_deterministically() {
    let mut accumulator = SKClassificationAccumulator::new();
    accumulator.observe(5, 3);
    accumulator.observe(2, 5);
    let matrix = accumulator
        .confusion_matrix(None)
        .expect("non-empty accumulator");
    assert_eq!(matrix.labels(), &[2, 3, 5]);
}

/// An explicit label space rejects observations outside it.
#[test]
fn explicit_label_space_rejects_unknown_labels() {
    let mut accumulator = SKClassificationAccumulator::new();
    accumulator.observe(2, 2);
    let error = accumulator
        .confusion_matrix(Some(&[0, 1]))
        .expect_err("label 2 is unknown");
    assert!(matches!(error, SKError::UnknownLabel { label: 2 }));
}

/// The accumulator's accuracy equals the whole-input accuracy.
#[test]
fn accumulator_accuracy_matches_whole_input() {
    let true_labels = [0_i64, 0, 1, 1];
    let predicted_labels = [0_i64, 1, 1, 1];
    let mut accumulator = SKClassificationAccumulator::new();
    for (true_label, predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        accumulator.observe(*true_label, *predicted_label);
    }
    let accuracy = accumulator
        .accuracy::<f64>()
        .expect("non-empty accumulator");
    let expected = crate::classification::sk_accuracy_score::<f64>(&true_labels, &predicted_labels)
        .expect("valid input");
    assert!((accuracy - expected).abs() < 1e-12);
}
