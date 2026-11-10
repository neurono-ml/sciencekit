//! Tests for the adjusted Rand index (spec `clustering-metrics`).

use ndarray::{Array1, array};

use super::sk_adjusted_rand_score;

#[test]
fn adjusted_rand_is_one_for_identical_labelings() {
    let labels = array![0_i64, 0, 1, 1, 2, 2];
    let score = sk_adjusted_rand_score(labels.view(), labels.view()).unwrap();
    assert!((score - 1.0).abs() < 1e-12);
}

#[test]
fn adjusted_rand_is_invariant_under_label_permutation() {
    let truth = array![0_i64, 0, 1, 1, 2, 2];
    let permuted = array![5_i64, 5, 9, 9, 2, 2];
    let original = sk_adjusted_rand_score(truth.view(), truth.view()).unwrap();
    let renamed = sk_adjusted_rand_score(truth.view(), permuted.view()).unwrap();
    assert!((renamed - original).abs() < 1e-12);
}

#[test]
fn adjusted_rand_matches_the_hand_computed_cross_fixture() {
    // a = [2, 2], b = [2, 2], every contingency cell equal to one:
    // numerator 0 - 4/6, denominator 2 - 4/6, so the index is -0.5.
    let truth = array![0_i64, 0, 1, 1];
    let predicted = array![0_i64, 1, 0, 1];
    let score = sk_adjusted_rand_score(truth.view(), predicted.view()).unwrap();
    assert!((score + 0.5).abs() < 1e-12, "expected -0.5, got {score}");
}

#[test]
fn adjusted_rand_is_near_zero_for_independent_labelings() {
    let number_of_rows = 120_usize;
    let truth: Array1<i64> = (0..number_of_rows)
        .map(|index| (index % 4) as i64)
        .collect();
    let predicted: Array1<i64> = (0..number_of_rows)
        .map(|index| ((index.wrapping_mul(1_103_515_245).wrapping_add(12_345) >> 16) % 5) as i64)
        .collect();
    let score = sk_adjusted_rand_score(truth.view(), predicted.view()).unwrap();
    assert!(
        score.abs() < 0.1,
        "independent labelings should score near zero, got {score}"
    );
}

#[test]
fn adjusted_rand_is_zero_at_the_chance_level() {
    // One trivial true labeling: the expected index equals the observed
    // agreement, so the chance-corrected value is exactly zero.
    let truth = array![0_i64, 0, 0, 0];
    let predicted = array![0_i64, 0, 1, 1];
    let score = sk_adjusted_rand_score(truth.view(), predicted.view()).unwrap();
    assert!(score.abs() < 1e-12, "expected zero, got {score}");
}
