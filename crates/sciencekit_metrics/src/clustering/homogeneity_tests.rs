//! Tests for homogeneity, completeness and V-measure (spec `clustering-metrics`).

use ndarray::array;

use super::{sk_completeness_score, sk_homogeneity_score, sk_v_measure_score};

#[test]
fn homogeneity_completeness_and_v_measure_are_one_for_identical_labelings() {
    let labels = array![0_i64, 0, 1, 1, 2, 2];
    assert!((sk_homogeneity_score(labels.view(), labels.view()).unwrap() - 1.0).abs() < 1e-12);
    assert!((sk_completeness_score(labels.view(), labels.view()).unwrap() - 1.0).abs() < 1e-12);
    assert!((sk_v_measure_score(labels.view(), labels.view()).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn homogeneity_completeness_and_v_measure_match_the_hand_computed_fixture() {
    // true = [0, 0, 1, 1], predicted = [0, 0, 0, 1].
    let truth = array![0_i64, 0, 1, 1];
    let predicted = array![0_i64, 0, 0, 1];
    assert!(
        (sk_homogeneity_score(truth.view(), predicted.view()).unwrap() - 0.31127812445913283).abs()
            < 1e-12
    );
    assert!(
        (sk_completeness_score(truth.view(), predicted.view()).unwrap() - 0.3836885465963443).abs()
            < 1e-12
    );
    assert!(
        (sk_v_measure_score(truth.view(), predicted.view()).unwrap() - 0.34371101848545077).abs()
            < 1e-12
    );
}

#[test]
fn homogeneity_scores_are_invariant_under_label_permutation() {
    let truth = array![0_i64, 0, 1, 1, 2, 2];
    let permuted = array![5_i64, 5, 9, 9, 2, 2];
    let expected = sk_v_measure_score(truth.view(), truth.view()).unwrap();
    let renamed = sk_v_measure_score(truth.view(), permuted.view()).unwrap();
    assert!((renamed - expected).abs() < 1e-12);
}

#[test]
fn homogeneity_handles_trivial_labelings() {
    let truth = array![0_i64, 0, 0, 0];
    let predicted = array![0_i64, 0, 1, 1];

    // A single true cluster is perfectly homogeneous but incomplete.
    assert!((sk_homogeneity_score(truth.view(), predicted.view()).unwrap() - 1.0).abs() < 1e-12);
    assert!(
        sk_completeness_score(truth.view(), predicted.view())
            .unwrap()
            .abs()
            < 1e-12
    );
    assert!(
        sk_v_measure_score(truth.view(), predicted.view())
            .unwrap()
            .abs()
            < 1e-12
    );

    // The reverse: incomplete is perfect when one predicted cluster holds all.
    let other = array![0_i64, 0, 0, 0];
    let fine = array![0_i64, 1, 2, 3];
    assert!(
        sk_homogeneity_score(fine.view(), other.view())
            .unwrap()
            .abs()
            < 1e-12
    );
    assert!((sk_completeness_score(fine.view(), other.view()).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn v_measure_is_one_when_both_labelings_are_trivial() {
    let truth = array![4_i64, 4, 4];
    let predicted = array![9_i64, 9, 9];
    assert!((sk_v_measure_score(truth.view(), predicted.view()).unwrap() - 1.0).abs() < 1e-12);
}
