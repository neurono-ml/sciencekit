//! TDD tests for binary ROC curves and AUC (spec `ranking-metrics`).
//!
//! Tie policy: equal scores share the average rank of the positions they
//! occupy (the Mann-Whitney rank-sum rule), so the AUC is deterministic.

use sciencekit_common::SKError;

use super::{sk_roc_auc_score, sk_roc_curve};

/// A perfect ranking places every positive above every negative.
#[test]
fn roc_perfect_ranking_has_unit_auc() {
    let scores = [0.1_f64, 0.2, 0.8, 0.9];
    let labels = [0_i64, 0, 1, 1];
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!((auc - 1.0).abs() < 1e-12);
}

/// A fully inverted ranking places every positive below every negative.
#[test]
fn roc_inverted_ranking_has_zero_auc() {
    let scores = [0.9_f64, 0.8, 0.2, 0.1];
    let labels = [0_i64, 0, 1, 1];
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!(auc.abs() < 1e-12);
}

/// A hand-computed fixture: positive 0.35 beats only negative 0.1 and the
/// positive 0.8 beats both negatives, so AUC = (1 + 2) / 4 = 0.75.
#[test]
fn roc_hand_computed_fixture() {
    let scores = [0.1_f64, 0.4, 0.35, 0.8];
    let labels = [0_i64, 0, 1, 1];
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!((auc - 0.75).abs() < 1e-12);
}

/// Ties share the average rank: 0.2 belongs to one positive and one negative
/// (ranks 2 and 3 average to 2.5), so the rank sum is 6.5 and
/// AUC = (6.5 − 3) / 4 = 0.875.
#[test]
fn roc_ties_use_average_ranks() {
    let scores = [0.1_f64, 0.2, 0.2, 0.3];
    let labels = [0_i64, 1, 0, 1];
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!((auc - 0.875).abs() < 1e-12);
}

/// A fully tied score gives every pair a half credit.
#[test]
fn roc_fully_tied_scores_have_half_auc() {
    let scores = [0.5_f64; 6];
    let labels = [1_i64, 0, 1, 0, 1, 0];
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!((auc - 0.5).abs() < 1e-12);
}

/// Interleaved positives and negatives rank at chance level.
#[test]
fn roc_random_interleaving_is_approximately_half() {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    for index in 0..200 {
        scores.push(index as f64);
        labels.push((index % 2) as i64);
    }
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!((auc - 0.5).abs() < 0.01, "chance AUC was {auc}");
}

/// Repeated runs over tied scores agree exactly.
#[test]
fn roc_ties_are_deterministic_across_runs() {
    let scores = [0.2_f64, 0.4, 0.2, 0.4, 0.2, 0.4];
    let labels = [0_i64, 1, 1, 0, 1, 0];
    let first = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    for _ in 0..32 {
        let again = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
        assert_eq!(first, again);
    }
}

/// The positive class is selected by its label, so flipping it inverts the AUC.
#[test]
fn roc_positive_label_selects_the_class() {
    let scores = [0.1_f64, 0.4, 0.35, 0.8];
    let labels = [0_i64, 0, 1, 1];
    let positive = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    let negative = sk_roc_auc_score(&scores, &labels, 0).expect("valid input");
    assert!((positive + negative - 1.0).abs() < 1e-12);
}

/// The curve walks the distinct scores in decreasing order.
#[test]
fn roc_curve_walks_distinct_scores_in_decreasing_order() {
    let scores = [0.1_f64, 0.4, 0.35, 0.8];
    let labels = [0_i64, 0, 1, 1];
    let curve = sk_roc_curve(&scores, &labels, 1).expect("valid input");
    assert_eq!(curve.thresholds(), &[0.8, 0.4, 0.35, 0.1]);
    assert_eq!(curve.true_positive_rates(), &[0.5, 0.5, 1.0, 1.0]);
    assert_eq!(curve.false_positive_rates(), &[0.0, 0.5, 0.5, 1.0]);
}

/// Empty input is rejected.
#[test]
fn roc_empty_input_is_rejected() {
    let error = sk_roc_auc_score::<f64>(&[], &[], 1).expect_err("empty input");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A score/label length mismatch is a shape mismatch.
#[test]
fn roc_length_mismatch_is_a_shape_mismatch() {
    let error = sk_roc_auc_score(&[0.1_f64, 0.2], &[0_i64], 1).expect_err("length mismatch");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A non-finite score is rejected at its position.
#[test]
fn roc_non_finite_score_is_rejected() {
    let error =
        sk_roc_auc_score(&[0.1_f64, f64::NAN], &[0_i64, 1], 1).expect_err("non-finite score");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}

/// A single-class problem has no defined AUC.
#[test]
fn roc_single_class_is_rejected() {
    let error = sk_roc_auc_score(&[0.1_f64, 0.2], &[1_i64, 1], 1).expect_err("one class");
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}
