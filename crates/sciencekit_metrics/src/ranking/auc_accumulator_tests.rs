//! TDD tests for the mergeable AUC accumulator (spec `ranking-metrics`).

use sciencekit_common::SKError;

use crate::ranking::sk_roc_auc_score;

use super::SKAucAccumulator;

/// Batched observation equals one whole-input AUC.
#[test]
fn auc_accumulator_batches_equal_whole_input() {
    let pairs = [
        (0.9_f64, 1_i64),
        (0.1, 0),
        (0.8, 1),
        (0.4, 0),
        (0.7, 1),
        (0.2, 0),
        (0.6, 1),
        (0.3, 0),
    ];
    let mut accumulator = SKAucAccumulator::new();
    for (score, label) in pairs.iter().take(3) {
        accumulator.observe(*score, *label);
    }
    for (score, label) in pairs.iter().skip(3) {
        accumulator.observe(*score, *label);
    }
    let scores: Vec<f64> = pairs.iter().map(|(score, _)| *score).collect();
    let labels: Vec<i64> = pairs.iter().map(|(_, label)| *label).collect();
    let expected = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    let accumulated = accumulator.finalize(1).expect("valid input");
    assert_eq!(accumulated, expected);
}

/// Merging two accumulators equals one accumulator over the concatenation.
#[test]
fn auc_accumulator_merge_equals_concatenation() {
    let first = [(0.9_f64, 1_i64), (0.4, 0), (0.8, 1), (0.2, 0)];
    let second = [(0.7_f64, 1_i64), (0.1, 0), (0.6, 1), (0.3, 0)];

    let mut merged = SKAucAccumulator::new();
    for (score, label) in first {
        merged.observe(score, label);
    }
    let mut other = SKAucAccumulator::new();
    for (score, label) in second {
        other.observe(score, label);
    }
    merged.merge(&other);

    let mut concatenated = SKAucAccumulator::new();
    for (score, label) in first.iter().chain(second.iter()) {
        concatenated.observe(*score, *label);
    }

    assert_eq!(
        merged.finalize(1).expect("valid input"),
        concatenated.finalize(1).expect("valid input")
    );
    assert_eq!(merged.number_of_observations(), 8);
}

/// The accumulator follows the same tie policy as the free function.
#[test]
fn auc_accumulator_handles_ties() {
    let pairs = [
        (0.2_f64, 1_i64),
        (0.2, 0),
        (0.2, 1),
        (0.4, 0),
        (0.6, 1),
        (0.6, 0),
    ];
    let mut accumulator = SKAucAccumulator::new();
    for (score, label) in pairs {
        accumulator.observe(score, label);
    }
    let scores = [0.2_f64, 0.2, 0.2, 0.4, 0.6, 0.6];
    let labels = [1_i64, 0, 1, 0, 1, 0];
    let expected = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert_eq!(accumulator.finalize(1).expect("valid input"), expected);
}

/// A fresh accumulator is empty and cannot finalize.
#[test]
fn auc_accumulator_empty_is_rejected() {
    let accumulator = SKAucAccumulator::<f64>::new();
    assert!(accumulator.is_empty());
    assert_eq!(accumulator.number_of_observations(), 0);
    let error = accumulator.finalize(1).expect_err("empty accumulator");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A non-finite observed score is rejected at finalization.
#[test]
fn auc_accumulator_rejects_non_finite_scores() {
    let mut accumulator = SKAucAccumulator::new();
    accumulator.observe(0.5_f64, 0);
    accumulator.observe(f64::NAN, 1);
    let error = accumulator.finalize(1).expect_err("non-finite score");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}

/// Observing one class only is rejected at finalization.
#[test]
fn auc_accumulator_rejects_single_class() {
    let mut accumulator = SKAucAccumulator::new();
    accumulator.observe(0.5_f64, 1);
    accumulator.observe(0.7_f64, 1);
    let error = accumulator.finalize(1).expect_err("one class only");
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}
