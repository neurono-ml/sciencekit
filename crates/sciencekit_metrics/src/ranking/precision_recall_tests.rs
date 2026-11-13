//! TDD tests for precision-recall curves and average precision (spec
//! `ranking-metrics`).
//!
//! Step interpolation: the curve sweeps distinct scores in decreasing order;
//! average precision is `Σ (R_n − R_{n−1}) · P_n` with `R_0 = 0`. Equal scores
//! are one threshold.

use sciencekit_common::SKError;

use super::{sk_average_precision_score, sk_precision_recall_curve};

/// A perfect ranking puts every positive first: average precision is one.
#[test]
fn precision_recall_perfect_ranking_has_unit_average_precision() {
    let scores = [0.1_f64, 0.2, 0.9, 0.95];
    let labels = [0_i64, 0, 1, 1];
    let average_precision = sk_average_precision_score(&scores, &labels, 1).expect("valid input");
    assert!((average_precision - 1.0).abs() < 1e-12);
}

/// Hand-computed fixture: thresholds 0.8, 0.4, 0.35 and 0.1 give
/// `0.5·1 + 0·0.5 + 0.5·(2/3) + 0·0.5 = 5/6`.
#[test]
fn precision_recall_hand_computed_average_precision() {
    let scores = [0.1_f64, 0.4, 0.35, 0.8];
    let labels = [0_i64, 0, 1, 1];
    let average_precision = sk_average_precision_score(&scores, &labels, 1).expect("valid input");
    assert!((average_precision - 5.0 / 6.0).abs() < 1e-12);
}

/// The curve walks distinct scores in decreasing order with grouped ties.
#[test]
fn precision_recall_curve_reports_grouped_thresholds() {
    let scores = [0.1_f64, 0.4, 0.35, 0.8];
    let labels = [0_i64, 0, 1, 1];
    let curve = sk_precision_recall_curve(&scores, &labels, 1).expect("valid input");
    assert_eq!(curve.thresholds(), &[0.8, 0.4, 0.35, 0.1]);
    assert_eq!(curve.recalls(), &[0.5, 0.5, 1.0, 1.0]);
    assert_eq!(curve.precisions(), &[1.0, 0.5, 2.0 / 3.0, 0.5]);
}

/// Fully tied scores form a single threshold at the base rate.
#[test]
fn precision_recall_ties_are_one_threshold() {
    let scores = [0.5_f64; 4];
    let labels = [1_i64, 0, 1, 0];
    let average_precision = sk_average_precision_score(&scores, &labels, 1).expect("valid input");
    assert!((average_precision - 0.5).abs() < 1e-12);
    let curve = sk_precision_recall_curve(&scores, &labels, 1).expect("valid input");
    assert_eq!(curve.thresholds(), &[0.5]);
    assert_eq!(curve.precisions(), &[0.5]);
    assert_eq!(curve.recalls(), &[1.0]);
}

/// An empty positive class is rejected.
#[test]
fn precision_recall_empty_positive_class_is_rejected() {
    let error =
        sk_average_precision_score(&[0.1_f64, 0.2], &[0_i64, 0], 1).expect_err("no positive class");
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}

/// Empty input is rejected.
#[test]
fn precision_recall_empty_input_is_rejected() {
    let error = sk_average_precision_score::<f64>(&[], &[], 1).expect_err("empty input");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A score/label length mismatch is a shape mismatch.
#[test]
fn precision_recall_length_mismatch_is_a_shape_mismatch() {
    let error =
        sk_average_precision_score(&[0.1_f64, 0.2], &[0_i64], 1).expect_err("length mismatch");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A non-finite score is rejected at its position.
#[test]
fn precision_recall_non_finite_score_is_rejected() {
    let error = sk_average_precision_score(&[0.1_f64, f64::NAN], &[1_i64, 0], 1)
        .expect_err("non-finite score");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}
