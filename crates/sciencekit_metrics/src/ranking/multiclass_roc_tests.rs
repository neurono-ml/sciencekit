//! TDD tests for multiclass one-vs-rest ROC/AUC (spec `ranking-metrics`).
//!
//! The documented aggregate is: each class is scored one-vs-rest with the
//! rank-sum AUC; macro is the unweighted mean, weighted is the support-weighted
//! mean and micro is one rank-sum AUC over every (row, class) pair.

use ndarray::{Array2, array};
use sciencekit_common::SKError;

use super::{SKRankingAveraging, sk_multiclass_roc_auc_score, sk_one_versus_rest_roc_auc_scores};

/// Perfect per-class probabilities give unit AUC under every averaging mode.
#[test]
fn multiclass_roc_perfect_probabilities_have_unit_auc() {
    let labels = [0_i64, 0, 1, 1, 2, 2];
    let scores = array![
        [0.9_f64, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.9, 0.0],
        [0.05, 0.8, 0.15],
        [0.0, 0.1, 0.9],
        [0.1, 0.0, 0.9],
    ];
    for averaging in [
        SKRankingAveraging::Macro,
        SKRankingAveraging::Weighted,
        SKRankingAveraging::Micro,
    ] {
        let auc =
            sk_multiclass_roc_auc_score(&scores.view(), &labels, averaging).expect("valid input");
        assert!(
            (auc - 1.0).abs() < 1e-12,
            "averaging {averaging:?} gave {auc}"
        );
    }
}

/// The fixture below has class AUCs 1, 2/3 and 1.
fn hand_computed_scores() -> Array2<f64> {
    array![
        [0.6_f64, 0.3, 0.1],
        [0.4, 0.5, 0.1],
        [0.2, 0.3, 0.5],
        [0.7, 0.8, 0.1],
    ]
}

/// Per-class AUCs are reported in ascending label order.
#[test]
fn multiclass_roc_per_class_aucs_are_hand_computed() {
    let labels = [0_i64, 1, 2, 0];
    let scores = hand_computed_scores();
    let per_class =
        sk_one_versus_rest_roc_auc_scores(&scores.view(), &labels).expect("valid input");
    assert_eq!(per_class.len(), 3);
    assert_eq!(per_class[0].0, 0);
    assert_eq!(per_class[1].0, 1);
    assert_eq!(per_class[2].0, 2);
    assert!((per_class[0].1 - 1.0).abs() < 1e-12);
    assert!((per_class[1].1 - 2.0 / 3.0).abs() < 1e-12);
    assert!((per_class[2].1 - 1.0).abs() < 1e-12);
}

/// Macro is the unweighted mean: (1 + 2/3 + 1) / 3 = 8/9.
#[test]
fn multiclass_roc_macro_average_is_unweighted_mean() {
    let labels = [0_i64, 1, 2, 0];
    let scores = hand_computed_scores();
    let auc = sk_multiclass_roc_auc_score(&scores.view(), &labels, SKRankingAveraging::Macro)
        .expect("valid input");
    assert!((auc - 8.0 / 9.0).abs() < 1e-12);
}

/// Weighted uses the support shares: (1·2 + 2/3·1 + 1·1) / 4 = 11/12.
#[test]
fn multiclass_roc_weighted_average_uses_support() {
    let labels = [0_i64, 1, 2, 0];
    let scores = hand_computed_scores();
    let auc = sk_multiclass_roc_auc_score(&scores.view(), &labels, SKRankingAveraging::Weighted)
        .expect("valid input");
    assert!((auc - 11.0 / 12.0).abs() < 1e-12);
}

/// Micro ranks the 12 flattened pairs at once: 28 winning pairs of 32 = 0.875.
#[test]
fn multiclass_roc_micro_average_ranks_all_pairs() {
    let labels = [0_i64, 1, 2, 0];
    let scores = hand_computed_scores();
    let auc = sk_multiclass_roc_auc_score(&scores.view(), &labels, SKRankingAveraging::Micro)
        .expect("valid input");
    assert!((auc - 0.875).abs() < 1e-12);
}

/// An empty label vector is rejected.
#[test]
fn multiclass_roc_empty_input_is_rejected() {
    let scores = Array2::<f64>::zeros((0, 0));
    let error = sk_multiclass_roc_auc_score(&scores.view(), &[], SKRankingAveraging::Macro)
        .expect_err("empty input");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A row/label mismatch is a shape mismatch.
#[test]
fn multiclass_roc_row_mismatch_is_a_shape_mismatch() {
    let scores = Array2::<f64>::zeros((2, 2));
    let error =
        sk_multiclass_roc_auc_score(&scores.view(), &[0_i64, 1, 2], SKRankingAveraging::Macro)
            .expect_err("two rows of scores, three labels");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The score matrix must carry one column per class.
#[test]
fn multiclass_roc_column_mismatch_is_a_shape_mismatch() {
    let scores = Array2::<f64>::zeros((2, 1));
    let error = sk_multiclass_roc_auc_score(&scores.view(), &[0_i64, 1], SKRankingAveraging::Macro)
        .expect_err("one column for two classes");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A non-finite score is rejected at its flattened position.
#[test]
fn multiclass_roc_non_finite_score_is_rejected() {
    let mut scores = Array2::<f64>::zeros((2, 2));
    scores[[1, 0]] = f64::INFINITY;
    let error = sk_multiclass_roc_auc_score(&scores.view(), &[0_i64, 1], SKRankingAveraging::Macro)
        .expect_err("non-finite score");
    assert!(matches!(error, SKError::NonFiniteValue { position: 2 }));
}
