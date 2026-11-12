//! Hand-computed tests for the classification impurities (spec `decision-trees`).

use super::{sk_entropy_impurity, sk_gini_impurity};
use std::f64::consts::LN_2;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, found {actual}"
    );
}

/// A pure node has zero Gini impurity and zero entropy.
#[test]
fn pure_node_has_zero_impurity() {
    assert_eq!(sk_gini_impurity(&[5.0, 0.0]), 0.0);
    assert_eq!(sk_entropy_impurity(&[5.0, 0.0]), 0.0);
}

/// An evenly split two-class node has Gini `1/2` and entropy `ln 2`.
#[test]
fn balanced_two_class_node_matches_the_closed_form() {
    assert_close(sk_gini_impurity(&[3.0, 3.0]), 0.5);
    assert_close(sk_entropy_impurity(&[3.0, 3.0]), LN_2);
}

/// A three-class distribution with counts `2, 1, 1` matches the hand-computed
/// Gini `0.625` and entropy `1.5 ln 2`.
#[test]
fn three_class_node_matches_hand_computation() {
    assert_close(sk_gini_impurity(&[2.0, 1.0, 1.0]), 0.625);
    assert_close(sk_entropy_impurity(&[2.0, 1.0, 1.0]), 1.5 * LN_2);
}

/// Weights scale the counts without changing the distribution: the same
/// proportions produce the same impurity.
#[test]
fn weighted_counts_preserve_the_impurity() {
    assert_close(
        sk_gini_impurity(&[2.0, 2.0]),
        sk_gini_impurity(&[20.0, 20.0]),
    );
}

/// Empty and all-zero count vectors are treated as pure.
#[test]
fn empty_or_zero_counts_are_pure() {
    let empty: [f64; 0] = [];
    assert_eq!(sk_gini_impurity(&empty), 0.0);
    assert_eq!(sk_entropy_impurity(&empty), 0.0);
    assert_eq!(sk_gini_impurity(&[0.0, 0.0]), 0.0);
    assert_eq!(sk_entropy_impurity(&[0.0, 0.0]), 0.0);
}

/// The measures are generic over both supported floats.
#[test]
fn impurity_accepts_either_float() {
    assert_eq!(sk_gini_impurity(&[1.0_f32, 1.0]), 0.5_f32);
    assert_eq!(sk_gini_impurity(&[1.0_f64, 1.0]), 0.5_f64);
}
