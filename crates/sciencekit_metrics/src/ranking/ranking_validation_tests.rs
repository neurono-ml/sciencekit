//! TDD validation tests for every ranking metric (spec `ranking-metrics`).
//!
//! The suite pins the rejected inputs: empty data, length mismatches, score
//! and probability range violations, single-class AUC and non-finite values.

use ndarray::{Array2, array};
use sciencekit_common::SKError;

use super::{
    SKAucAccumulator, SKRankingAveraging, sk_average_precision_score, sk_balanced_accuracy_score,
    sk_binary_log_loss, sk_log_loss, sk_matthews_correlation_coefficient,
    sk_multiclass_roc_auc_score, sk_precision_recall_curve, sk_roc_auc_score, sk_roc_curve,
};

/// Every score-based metric rejects empty input.
#[test]
fn ranking_validation_empty_input_is_rejected_everywhere() {
    assert!(matches!(
        sk_roc_auc_score::<f64>(&[], &[], 1),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        sk_roc_curve::<f64>(&[], &[], 1),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        sk_average_precision_score::<f64>(&[], &[], 1),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        sk_precision_recall_curve::<f64>(&[], &[], 1),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        sk_balanced_accuracy_score::<f64>(&[], &[]),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        sk_matthews_correlation_coefficient::<f64>(&[], &[]),
        Err(SKError::EmptyInput { .. })
    ));
    let probabilities = Array2::<f64>::zeros((0, 2));
    assert!(matches!(
        sk_log_loss(&[], probabilities.view(), None),
        Err(SKError::EmptyInput { .. })
    ));
    assert!(matches!(
        sk_binary_log_loss::<f64>(&[], array![].view(), 1),
        Err(SKError::EmptyInput { .. })
    ));
}

/// Every length mismatch is a shape mismatch.
#[test]
fn ranking_validation_length_mismatch_is_a_shape_mismatch() {
    assert!(matches!(
        sk_roc_auc_score(&[0.1_f64, 0.2], &[0_i64], 1),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        sk_average_precision_score(&[0.1_f64, 0.2], &[0_i64], 1),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        sk_balanced_accuracy_score::<f64>(&[0_i64, 1], &[0_i64]),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        sk_matthews_correlation_coefficient::<f64>(&[0_i64, 1], &[0_i64]),
        Err(SKError::ShapeMismatch { .. })
    ));
    let probabilities = Array2::<f64>::zeros((2, 2));
    assert!(matches!(
        sk_log_loss(&[0_i64], probabilities.view(), None),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        sk_binary_log_loss(&[0_i64], array![0.5_f64, 0.5].view(), 1),
        Err(SKError::ShapeMismatch { .. })
    ));
    let scores = Array2::<f64>::zeros((2, 3));
    assert!(matches!(
        sk_multiclass_roc_auc_score(&scores.view(), &[0_i64, 1], SKRankingAveraging::Macro),
        Err(SKError::ShapeMismatch { .. })
    ));
}

/// Probabilities outside the closed unit interval are rejected.
#[test]
fn ranking_validation_probability_range_is_enforced() {
    let above = array![[1.1_f64, -0.1]];
    assert!(matches!(
        sk_log_loss(&[0_i64], above.view(), None),
        Err(SKError::ProbabilityOutOfRange { position: 0 })
    ));
    let below = array![[0.5_f64, 1.2]];
    assert!(matches!(
        sk_log_loss(&[0_i64], below.view(), None),
        Err(SKError::ProbabilityOutOfRange { position: 1 })
    ));
    assert!(matches!(
        sk_binary_log_loss(&[1_i64], array![-0.5_f64].view(), 1),
        Err(SKError::ProbabilityOutOfRange { position: 0 })
    ));
}

/// A single-class ranking has no defined AUC.
#[test]
fn ranking_validation_single_class_auc_is_rejected() {
    assert!(matches!(
        sk_roc_auc_score(&[0.1_f64, 0.2], &[1_i64, 1], 1),
        Err(SKError::UnsupportedOperation { .. })
    ));
    assert!(matches!(
        sk_average_precision_score(&[0.1_f64, 0.2], &[0_i64, 0], 1),
        Err(SKError::UnsupportedOperation { .. })
    ));
    let scores = array![[0.9_f64], [0.1]];
    assert!(matches!(
        sk_multiclass_roc_auc_score(&scores.view(), &[0_i64, 0], SKRankingAveraging::Macro),
        Err(SKError::UnsupportedOperation { .. })
    ));
}

/// Non-finite scores and probabilities are rejected at their position.
#[test]
fn ranking_validation_non_finite_values_are_rejected() {
    assert!(matches!(
        sk_roc_auc_score(&[0.1_f64, f64::NAN], &[0_i64, 1], 1),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
    assert!(matches!(
        sk_average_precision_score(&[0.1_f64, f64::INFINITY], &[0_i64, 1], 1),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
    let probabilities = array![[0.5_f64, f64::NAN]];
    assert!(matches!(
        sk_log_loss(&[0_i64], probabilities.view(), None),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
    let scores = array![[0.5_f64, 0.5], [f64::NAN, 0.5]];
    assert!(matches!(
        sk_multiclass_roc_auc_score(&scores.view(), &[0_i64, 1], SKRankingAveraging::Macro),
        Err(SKError::NonFiniteValue { position: 2 })
    ));
}

/// The accumulator defers validation until finalization.
#[test]
fn ranking_validation_accumulator_rejects_invalid_state() {
    let accumulator = SKAucAccumulator::<f64>::new();
    assert!(matches!(
        accumulator.finalize(1),
        Err(SKError::EmptyInput { .. })
    ));
    let mut accumulator = SKAucAccumulator::new();
    accumulator.observe(f64::INFINITY, 1);
    accumulator.observe(0.5, 0);
    assert!(matches!(
        accumulator.finalize(1),
        Err(SKError::NonFiniteValue { position: 0 })
    ));
}

/// A multiclass score matrix needs one column per observed class.
#[test]
fn ranking_validation_multiclass_column_count_is_enforced() {
    let scores = array![[0.5_f64, 0.5, 0.0]];
    assert!(matches!(
        sk_multiclass_roc_auc_score(&scores.view(), &[0_i64, 1], SKRankingAveraging::Macro),
        Err(SKError::ShapeMismatch { .. })
    ));
}
