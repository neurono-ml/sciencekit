//! Tests for normalized mutual information (spec `clustering-metrics`).

use ndarray::array;

use super::{SKMutualInformationNormalization, sk_normalized_mutual_information_score};

#[test]
fn nmi_is_one_for_identical_labelings_in_every_normalization() {
    let labels = array![0_i64, 0, 1, 1, 2, 2];
    for normalization in [
        SKMutualInformationNormalization::Arithmetic,
        SKMutualInformationNormalization::Geometric,
        SKMutualInformationNormalization::Maximum,
    ] {
        let score =
            sk_normalized_mutual_information_score(labels.view(), labels.view(), normalization)
                .unwrap();
        assert!(
            (score - 1.0).abs() < 1e-12,
            "{normalization:?} scored {score}"
        );
    }
}

#[test]
fn nmi_is_invariant_under_label_permutation() {
    let truth = array![0_i64, 0, 1, 1, 2, 2];
    let permuted = array![7_i64, 7, 4, 4, 1, 1];
    for normalization in [
        SKMutualInformationNormalization::Arithmetic,
        SKMutualInformationNormalization::Geometric,
        SKMutualInformationNormalization::Maximum,
    ] {
        let original =
            sk_normalized_mutual_information_score(truth.view(), truth.view(), normalization)
                .unwrap();
        let renamed =
            sk_normalized_mutual_information_score(truth.view(), permuted.view(), normalization)
                .unwrap();
        assert!((renamed - original).abs() < 1e-12);
    }
}

#[test]
fn nmi_matches_the_hand_computed_normalization_variants() {
    // true = [0, 0, 1, 1], predicted = [0, 0, 0, 1]:
    // H(true) = ln 2, H(predicted) = 0.562335, MI = 0.215762.
    let truth = array![0_i64, 0, 1, 1];
    let predicted = array![0_i64, 0, 0, 1];

    let arithmetic = sk_normalized_mutual_information_score(
        truth.view(),
        predicted.view(),
        SKMutualInformationNormalization::Arithmetic,
    )
    .unwrap();
    let geometric = sk_normalized_mutual_information_score(
        truth.view(),
        predicted.view(),
        SKMutualInformationNormalization::Geometric,
    )
    .unwrap();
    let maximum = sk_normalized_mutual_information_score(
        truth.view(),
        predicted.view(),
        SKMutualInformationNormalization::Maximum,
    )
    .unwrap();

    assert!((arithmetic - 0.3437110184854508).abs() < 1e-12);
    assert!((geometric - 0.3455920299442113).abs() < 1e-12);
    assert!((maximum - 0.31127812445913283).abs() < 1e-12);
}

#[test]
fn nmi_is_zero_for_independent_labelings() {
    let truth = array![0_i64, 0, 1, 1];
    let predicted = array![0_i64, 1, 0, 1];
    for normalization in [
        SKMutualInformationNormalization::Arithmetic,
        SKMutualInformationNormalization::Geometric,
        SKMutualInformationNormalization::Maximum,
    ] {
        let score =
            sk_normalized_mutual_information_score(truth.view(), predicted.view(), normalization)
                .unwrap();
        assert!(score.abs() < 1e-12, "{normalization:?} scored {score}");
    }
}

#[test]
fn nmi_is_one_when_both_labelings_are_trivial() {
    let truth = array![3_i64, 3, 3, 3];
    let predicted = array![8_i64, 8, 8, 8];
    let score = sk_normalized_mutual_information_score(
        truth.view(),
        predicted.view(),
        SKMutualInformationNormalization::Arithmetic,
    )
    .unwrap();
    assert!((score - 1.0).abs() < 1e-12);
}

#[test]
fn nmi_is_zero_when_only_one_labeling_is_trivial() {
    let truth = array![0_i64, 0, 0, 0];
    let predicted = array![0_i64, 0, 1, 1];
    let score = sk_normalized_mutual_information_score(
        truth.view(),
        predicted.view(),
        SKMutualInformationNormalization::Arithmetic,
    )
    .unwrap();
    assert!(score.abs() < 1e-12);
}
