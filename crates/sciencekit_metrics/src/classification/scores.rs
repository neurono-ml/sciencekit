//! Whole-input precision, recall and F1 score functions.

use sciencekit_common::{SKError, SKFloat};

use super::SKConfusionMatrix;
use super::{
    SKAggregatedScores, SKAveraging, sk_aggregate_class_scores, sk_confusion_matrix,
    sk_per_class_precision_recall_f1,
};

/// The aggregated precision of matched label pairs.
pub fn sk_precision_score<F: SKFloat>(
    true_labels: &[i64],
    predicted_labels: &[i64],
    explicit_labels: Option<&[i64]>,
    averaging: SKAveraging,
) -> Result<F, SKError> {
    Ok(sk_aggregated_scores(
        true_labels,
        predicted_labels,
        explicit_labels,
        averaging,
    )?
    .precision())
}

/// The aggregated recall of matched label pairs.
pub fn sk_recall_score<F: SKFloat>(
    true_labels: &[i64],
    predicted_labels: &[i64],
    explicit_labels: Option<&[i64]>,
    averaging: SKAveraging,
) -> Result<F, SKError> {
    Ok(sk_aggregated_scores(
        true_labels,
        predicted_labels,
        explicit_labels,
        averaging,
    )?
    .recall())
}

/// The aggregated F1 of matched label pairs.
pub fn sk_f1_score<F: SKFloat>(
    true_labels: &[i64],
    predicted_labels: &[i64],
    explicit_labels: Option<&[i64]>,
    averaging: SKAveraging,
) -> Result<F, SKError> {
    Ok(sk_aggregated_scores(
        true_labels,
        predicted_labels,
        explicit_labels,
        averaging,
    )?
    .f1())
}

/// Build the matrix, the per-class scores and the aggregate once.
fn sk_aggregated_scores<F: SKFloat>(
    true_labels: &[i64],
    predicted_labels: &[i64],
    explicit_labels: Option<&[i64]>,
    averaging: SKAveraging,
) -> Result<SKAggregatedScores<F>, SKError> {
    let matrix: SKConfusionMatrix =
        sk_confusion_matrix(true_labels, predicted_labels, explicit_labels)?;
    let scores = sk_per_class_precision_recall_f1::<F>(&matrix);
    sk_aggregate_class_scores(&matrix, &scores, averaging)
}
