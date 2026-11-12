//! f32/f64 parity of the ranking metrics.

use crate::ranking::{sk_average_precision_score, sk_log_loss, sk_roc_auc_score};

/// f32 and f64 AUC and average precision agree within single precision.
#[test]
fn ranking_acceptance_f32_matches_f64() {
    let scores_f64 = [0.1_f64, 0.4, 0.35, 0.8, 0.2, 0.6];
    let labels = [0_i64, 0, 1, 1, 0, 1];
    let scores_f32: Vec<f32> = scores_f64.iter().map(|score| *score as f32).collect();
    let auc = sk_roc_auc_score(&scores_f64, &labels, 1).expect("valid input");
    let auc_f32 = sk_roc_auc_score(&scores_f32, &labels, 1).expect("valid input");
    assert!((auc - auc_f32 as f64).abs() < 1e-5);

    let average_precision = sk_average_precision_score(&scores_f64, &labels, 1).expect("valid");
    let average_precision_f32 = sk_average_precision_score(&scores_f32, &labels, 1).expect("valid");
    assert!((average_precision - average_precision_f32 as f64).abs() < 1e-5);
}

/// f32 log loss matches f64 within single precision.
#[test]
fn ranking_acceptance_f32_log_loss_matches_f64() {
    let labels = [0_i64, 1, 1];
    let probabilities = ndarray::array![[0.7_f64, 0.3], [0.4, 0.6], [0.2, 0.8]];
    let probabilities_f32 = probabilities.mapv(|value| value as f32);
    let loss = sk_log_loss(&labels, probabilities.view(), None).expect("valid input");
    let loss_f32 = sk_log_loss(&labels, probabilities_f32.view(), None).expect("valid input");
    assert!((loss - loss_f32 as f64).abs() < 1e-5);
}
