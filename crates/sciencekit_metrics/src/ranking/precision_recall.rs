//! Precision-recall curves and average precision.
//!
//! The curve sweeps distinct scores in decreasing order and groups equal scores
//! into one threshold. Average precision is the step-wise area
//! `Σ (R_n − R_{n−1}) · P_n` with `R_0 = 0` — the same convention scikit-learn
//! documents for `average_precision_score`.

use sciencekit_common::{SKError, SKFloat};

use super::inputs::{
    sk_ascending_score_order, sk_require_positive_class, sk_validate_score_label_inputs,
};

/// A binary precision-recall curve over the distinct scores.
#[derive(Debug, Clone, PartialEq)]
pub struct SKPrecisionRecallCurve<F> {
    precisions: Vec<F>,
    recalls: Vec<F>,
    thresholds: Vec<F>,
}

impl<F> SKPrecisionRecallCurve<F> {
    /// The precision at each threshold.
    pub fn precisions(&self) -> &[F] {
        &self.precisions
    }

    /// The recall at each threshold.
    pub fn recalls(&self) -> &[F] {
        &self.recalls
    }

    /// The distinct scores, in decreasing order.
    pub fn thresholds(&self) -> &[F] {
        &self.thresholds
    }
}

/// Compute the binary precision-recall curve of the given scores and labels.
pub fn sk_precision_recall_curve<F: SKFloat>(
    scores: &[F],
    labels: &[i64],
    positive_label: i64,
) -> Result<SKPrecisionRecallCurve<F>, SKError> {
    sk_validate_score_label_inputs(scores, labels, "precision_recall_curve")?;
    let positives = sk_require_positive_class(labels, positive_label, "precision_recall_curve")?;
    let positives = num_traits::cast::<usize, F>(positives).expect("counts fit in the scalar");

    let mut order = sk_ascending_score_order(scores);
    order.reverse();
    let mut precisions = Vec::new();
    let mut recalls = Vec::new();
    let mut thresholds = Vec::new();
    let mut true_positives = 0_usize;
    let mut false_positives = 0_usize;

    let mut start = 0;
    while start < order.len() {
        let threshold = scores[order[start]];
        let mut end = start;
        while end < order.len() && scores[order[end]] == threshold {
            if labels[order[end]] == positive_label {
                true_positives += 1;
            } else {
                false_positives += 1;
            }
            end += 1;
        }
        let selected = true_positives + false_positives;
        let selected = num_traits::cast::<usize, F>(selected).expect("counts fit in the scalar");
        let true_positives =
            num_traits::cast::<usize, F>(true_positives).expect("counts fit in the scalar");
        precisions.push(true_positives / selected);
        recalls.push(true_positives / positives);
        thresholds.push(threshold);
        start = end;
    }

    Ok(SKPrecisionRecallCurve {
        precisions,
        recalls,
        thresholds,
    })
}

/// Compute the binary average precision of the given scores and labels.
pub fn sk_average_precision_score<F: SKFloat>(
    scores: &[F],
    labels: &[i64],
    positive_label: i64,
) -> Result<F, SKError> {
    let curve = sk_precision_recall_curve(scores, labels, positive_label)?;
    let mut previous_recall = F::zero();
    let mut average_precision = F::zero();
    for (precision, recall) in curve.precisions.iter().zip(curve.recalls.iter()) {
        average_precision = average_precision + (*recall - previous_recall) * *precision;
        previous_recall = *recall;
    }
    Ok(average_precision)
}
