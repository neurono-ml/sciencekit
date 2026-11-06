//! TDD tests for score aggregation (spec `evaluation-metrics`).

use sciencekit_common::SKError;

use super::{
    SKAveraging, SKConfusionMatrix, sk_accuracy_score, sk_aggregate_class_scores,
    sk_confusion_matrix, sk_per_class_precision_recall_f1,
};

/// A matrix with unequal class supports, used by the averaging tests.
fn unbalanced_matrix() -> SKConfusionMatrix {
    // Truth: [0,0,0,0,1], predictions: [0,0,0,1,1].
    sk_confusion_matrix(&[0, 0, 0, 0, 1], &[0, 0, 0, 1, 1], None).expect("valid input")
}

/// Macro averaging weights every class equally.
#[test]
fn macro_averaging_weights_classes_equally() {
    let matrix = unbalanced_matrix();
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    let aggregated =
        sk_aggregate_class_scores(&matrix, &scores, SKAveraging::Macro).expect("valid");
    // precision: (1 + 0.5) / 2.
    assert!((aggregated.precision() - 0.75).abs() < 1e-12);
    // recall: (0.75 + 1) / 2.
    assert!((aggregated.recall() - 0.875).abs() < 1e-12);
}

/// Micro-averaged F1 equals accuracy on a single-label multiclass problem.
#[test]
fn micro_averaging_equals_accuracy() {
    let matrix = unbalanced_matrix();
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    let aggregated =
        sk_aggregate_class_scores(&matrix, &scores, SKAveraging::Micro).expect("valid");
    let accuracy =
        sk_accuracy_score::<f64>(&[0, 0, 0, 0, 1], &[0, 0, 0, 1, 1]).expect("valid input");
    assert!((aggregated.precision() - accuracy).abs() < 1e-12);
    assert!((aggregated.recall() - accuracy).abs() < 1e-12);
    assert!((aggregated.f1() - accuracy).abs() < 1e-12);
}

/// Weighted averaging follows class support.
#[test]
fn weighted_averaging_follows_support() {
    let matrix = unbalanced_matrix();
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    let aggregated =
        sk_aggregate_class_scores(&matrix, &scores, SKAveraging::Weighted).expect("valid");
    // precision: (1 · 4 + 0.5 · 1) / 5.
    assert!((aggregated.precision() - 0.9).abs() < 1e-12);
    // recall: (0.75 · 4 + 1 · 1) / 5.
    assert!((aggregated.recall() - 0.8).abs() < 1e-12);
}

/// Binary averaging reports the chosen class's scores.
#[test]
fn binary_averaging_reports_the_positive_class() {
    let matrix = unbalanced_matrix();
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    let aggregated =
        sk_aggregate_class_scores(&matrix, &scores, SKAveraging::Binary { positive_label: 1 })
            .expect("valid");
    assert!((aggregated.precision() - 0.5).abs() < 1e-12);
    assert!((aggregated.recall() - 1.0).abs() < 1e-12);
}

/// A positive label outside the space is rejected.
#[test]
fn binary_averaging_rejects_an_unknown_positive_label() {
    let matrix = unbalanced_matrix();
    let scores = sk_per_class_precision_recall_f1::<f64>(&matrix);
    let error =
        sk_aggregate_class_scores(&matrix, &scores, SKAveraging::Binary { positive_label: 9 })
            .expect_err("label 9 is unknown");
    assert!(matches!(error, SKError::UnknownLabel { label: 9 }));
}
