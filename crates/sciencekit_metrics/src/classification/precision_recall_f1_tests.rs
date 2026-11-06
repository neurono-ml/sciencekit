//! TDD tests for the per-class precision, recall and F1 metrics
//! (spec `evaluation-metrics`).

use super::{SKConfusionMatrix, sk_confusion_matrix, sk_per_class_precision_recall_f1};

/// Per-class scores follow the standard definitions on a known matrix.
#[test]
fn per_class_scores_match_a_known_example() {
    let matrix: SKConfusionMatrix =
        sk_confusion_matrix(&[0, 0, 1, 1], &[0, 0, 0, 1], None).expect("valid input");
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    assert_eq!(scores.precision().len(), 2);
    assert!((scores.precision()[0] - 2.0 / 3.0).abs() < 1e-12);
    assert!((scores.recall()[0] - 1.0).abs() < 1e-12);
    assert!((scores.f1()[0] - 0.8).abs() < 1e-12);
    assert!((scores.precision()[1] - 1.0).abs() < 1e-12);
    assert!((scores.recall()[1] - 0.5).abs() < 1e-12);
    assert!((scores.f1()[1] - 2.0 / 3.0).abs() < 1e-12);
}

/// A class present in the truth but never predicted has zero precision.
#[test]
fn absent_predicted_class_has_zero_precision() {
    let matrix = sk_confusion_matrix(&[0, 0], &[0, 0], Some(&[0, 1])).expect("valid input");
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    assert!(scores.precision()[1].abs() < 1e-12);
    assert!(scores.recall()[1].abs() < 1e-12);
    assert!(scores.f1()[1].abs() < 1e-12);
}

/// A class predicted but never present in the truth has zero recall.
#[test]
fn absent_true_class_has_zero_recall() {
    let matrix = sk_confusion_matrix(&[0, 0], &[1, 1], Some(&[0, 1])).expect("valid input");
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    assert!(scores.precision()[0].abs() < 1e-12);
    assert!(scores.recall()[0].abs() < 1e-12);
    assert!(scores.precision()[1].abs() < 1e-12);
    assert!(scores.recall()[1].abs() < 1e-12);
}

/// A resolved class with no observation at all scores zero everywhere.
#[test]
fn unobserved_class_scores_zero_without_nan() {
    let matrix = sk_confusion_matrix(&[0], &[0], Some(&[0, 5])).expect("valid input");
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    assert_eq!(scores.precision().len(), 2);
    assert!(scores.precision()[1].is_finite());
    assert!(scores.recall()[1].is_finite());
    assert!(scores.f1()[1].is_finite());
    assert!(scores.f1()[1].abs() < 1e-12);
}
