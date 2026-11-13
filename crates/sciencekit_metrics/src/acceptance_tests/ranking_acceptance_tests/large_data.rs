//! Large-data acceptance of the ranking metrics.

use crate::ranking::{
    SKAucAccumulator, sk_average_precision_score, sk_balanced_accuracy_score, sk_log_loss,
    sk_matthews_correlation_coefficient, sk_roc_auc_score,
};

use super::fixtures::{OBSERVATIONS, large_scores_and_labels};

/// Large, perfectly separable data scores one on every rank-based metric.
#[test]
fn ranking_acceptance_large_data_scores_one() {
    let (scores, labels) = large_scores_and_labels();
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!((auc - 1.0).abs() < 1e-12);
    let average_precision = sk_average_precision_score(&scores, &labels, 1).expect("valid input");
    assert!((average_precision - 1.0).abs() < 1e-12);
    let predicted: Vec<i64> = scores
        .iter()
        .map(|score| if *score >= 0.5 { 1 } else { 0 })
        .collect();
    assert!((sk_balanced_accuracy_score::<f64>(&labels, &predicted).unwrap() - 1.0).abs() < 1e-12);
    assert!(
        (sk_matthews_correlation_coefficient::<f64>(&labels, &predicted).unwrap() - 1.0).abs()
            < 1e-12
    );
}

/// A large probability matrix produces a finite log loss.
#[test]
fn ranking_acceptance_large_data_log_loss_is_finite() {
    let (scores, labels) = large_scores_and_labels();
    let mut probabilities = ndarray::Array2::<f64>::zeros((OBSERVATIONS, 2));
    for (row, score) in scores.iter().enumerate() {
        let positive = 0.6 + 0.3 * score;
        probabilities[[row, 0]] = 1.0 - positive;
        probabilities[[row, 1]] = positive;
    }
    let loss = sk_log_loss(&labels, probabilities.view(), None).expect("valid input");
    assert!(loss.is_finite() && loss > 0.0 && loss < 1.0);
}

/// A large accumulator in batches equals the whole-input AUC.
#[test]
fn ranking_acceptance_large_accumulator_matches_whole_input() {
    let (scores, labels) = large_scores_and_labels();
    let mut accumulator = SKAucAccumulator::new();
    for (score, label) in scores.iter().zip(labels.iter()) {
        accumulator.observe(*score, *label);
    }
    let accumulated = accumulator.finalize(1).expect("valid input");
    let whole = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert_eq!(accumulated, whole);
}
