//! TDD tests for balanced accuracy and the Matthews correlation coefficient
//! (spec `ranking-metrics`).
//!
//! Balanced accuracy is the mean per-class recall over classes present in the
//! true labels. MCC uses the generalised confusion-matrix formula with a
//! zero-denominator policy of zero.

use sciencekit_common::SKError;

use super::{sk_balanced_accuracy_score, sk_matthews_correlation_coefficient};

/// Balanced classes predicted perfectly score one on both metrics.
#[test]
fn balanced_mcc_perfect_balanced_predictions_score_one() {
    let truth = [0_i64, 0, 1, 1];
    let predicted = [0_i64, 0, 1, 1];
    assert!((sk_balanced_accuracy_score::<f64>(&truth, &predicted).unwrap() - 1.0).abs() < 1e-12);
    assert!(
        (sk_matthews_correlation_coefficient::<f64>(&truth, &predicted).unwrap() - 1.0).abs()
            < 1e-12
    );
}

/// Predicting only the majority class on an imbalanced problem leaves balanced
/// accuracy at 0.5 and MCC at zero.
#[test]
fn balanced_mcc_majority_only_prediction() {
    let truth = [0_i64, 0, 0, 0, 1, 1];
    let predicted = [0_i64; 6];
    let balanced = sk_balanced_accuracy_score::<f64>(&truth, &predicted).expect("valid input");
    assert!((balanced - 0.5).abs() < 1e-12);
    let mcc = sk_matthews_correlation_coefficient::<f64>(&truth, &predicted).expect("valid input");
    assert!(mcc.abs() < 1e-12);
}

/// Hand-computed MCC: TP=1, TN=2, FP=0, FN=1 gives 2 / √12.
#[test]
fn balanced_mcc_binary_hand_computed() {
    let truth = [0_i64, 0, 1, 1];
    let predicted = [0_i64, 0, 1, 0];
    let mcc = sk_matthews_correlation_coefficient::<f64>(&truth, &predicted).expect("valid input");
    assert!((mcc - 2.0 / 12.0_f64.sqrt()).abs() < 1e-12);
}

/// Hand-computed multiclass fixture: a cyclic shift gives MCC = 0.25 and
/// balanced accuracy = 0.5.
#[test]
fn balanced_mcc_multiclass_hand_computed() {
    let truth = [0_i64, 0, 1, 1, 2, 2];
    let predicted = [0_i64, 1, 1, 2, 2, 0];
    let mcc = sk_matthews_correlation_coefficient::<f64>(&truth, &predicted).expect("valid input");
    assert!((mcc - 0.25).abs() < 1e-12);
    let balanced = sk_balanced_accuracy_score::<f64>(&truth, &predicted).expect("valid input");
    assert!((balanced - 0.5).abs() < 1e-12);
}

/// Balanced accuracy is the mean of the per-class recalls.
#[test]
fn balanced_accuracy_is_the_mean_recall() {
    let truth = [0_i64, 0, 0, 1, 1];
    let predicted = [0_i64, 0, 1, 1, 0];
    // Recall class 0 = 2/3 and class 1 = 1/2.
    let balanced = sk_balanced_accuracy_score::<f64>(&truth, &predicted).expect("valid input");
    assert!((balanced - (2.0 / 3.0 + 0.5) / 2.0).abs() < 1e-12);
}

/// A class present only in the predictions does not contribute a support term.
#[test]
fn balanced_accuracy_ignores_absent_true_classes() {
    let truth = [0_i64, 0, 0];
    let predicted = [0_i64, 0, 2];
    let balanced = sk_balanced_accuracy_score::<f64>(&truth, &predicted).expect("valid input");
    assert!((balanced - 2.0 / 3.0).abs() < 1e-12);
}

/// Empty input is rejected.
#[test]
fn balanced_mcc_empty_input_is_rejected() {
    let error = sk_balanced_accuracy_score::<f64>(&[], &[]).expect_err("empty input");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A length mismatch is a shape mismatch.
#[test]
fn balanced_mcc_length_mismatch_is_a_shape_mismatch() {
    let error =
        sk_balanced_accuracy_score::<f64>(&[0_i64, 1], &[0_i64]).expect_err("length mismatch");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
