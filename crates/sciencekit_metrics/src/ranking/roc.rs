//! Binary ROC curves and the rank-sum (Mann–Whitney) ROC-AUC.
//!
//! Ties are the subtle part: equal scores share the average of the ranks they
//! occupy, which is exactly the Mann–Whitney definition of AUC and matches
//! scikit-learn's `roc_auc_score` for tied inputs.

use sciencekit_common::{SKError, SKFloat};

use super::inputs::{
    sk_ascending_score_order, sk_binary_class_counts, sk_validate_score_label_inputs,
};

/// A binary ROC curve: the false- and true-positive rates at every distinct
/// score, walked in decreasing score order.
#[derive(Debug, Clone, PartialEq)]
pub struct SKRocCurve<F> {
    false_positive_rates: Vec<F>,
    true_positive_rates: Vec<F>,
    thresholds: Vec<F>,
}

impl<F> SKRocCurve<F> {
    /// The false-positive rate at each threshold.
    pub fn false_positive_rates(&self) -> &[F] {
        &self.false_positive_rates
    }

    /// The true-positive rate at each threshold.
    pub fn true_positive_rates(&self) -> &[F] {
        &self.true_positive_rates
    }

    /// The distinct scores, in decreasing order.
    pub fn thresholds(&self) -> &[F] {
        &self.thresholds
    }
}

/// Compute the binary ROC curve of the given scores and labels.
///
/// A label equal to `positive_label` is positive; every other label is
/// negative. Equal scores are grouped into one threshold.
pub fn sk_roc_curve<F: SKFloat>(
    scores: &[F],
    labels: &[i64],
    positive_label: i64,
) -> Result<SKRocCurve<F>, SKError> {
    sk_validate_score_label_inputs(scores, labels, "roc_curve")?;
    let (positives, negatives) = sk_binary_class_counts(labels, positive_label, "roc_curve")?;
    let positives = num_traits::cast::<usize, F>(positives).expect("counts fit in the scalar");
    let negatives = num_traits::cast::<usize, F>(negatives).expect("counts fit in the scalar");

    let mut order = sk_ascending_score_order(scores);
    order.reverse();
    let mut false_positive_rates = Vec::new();
    let mut true_positive_rates = Vec::new();
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
        let true_positives =
            num_traits::cast::<usize, F>(true_positives).expect("counts fit in the scalar");
        let false_positives =
            num_traits::cast::<usize, F>(false_positives).expect("counts fit in the scalar");
        false_positive_rates.push(false_positives / negatives);
        true_positive_rates.push(true_positives / positives);
        thresholds.push(threshold);
        start = end;
    }

    Ok(SKRocCurve {
        false_positive_rates,
        true_positive_rates,
        thresholds,
    })
}

/// Compute the binary ROC-AUC of the given scores and labels.
///
/// The value is the Mann–Whitney statistic over the score ranking: equal
/// scores share the average rank of the positions they occupy.
pub fn sk_roc_auc_score<F: SKFloat>(
    scores: &[F],
    labels: &[i64],
    positive_label: i64,
) -> Result<F, SKError> {
    sk_validate_score_label_inputs(scores, labels, "roc_auc_score")?;
    sk_binary_class_counts(labels, positive_label, "roc_auc_score")?;
    Ok(sk_rank_auc(scores, labels, positive_label))
}

/// The rank-sum AUC over already-validated scores and labels.
pub(super) fn sk_rank_auc<F: SKFloat>(scores: &[F], labels: &[i64], positive_label: i64) -> F {
    let order = sk_ascending_score_order(scores);
    let positive_count = labels
        .iter()
        .filter(|label| **label == positive_label)
        .count();
    let negative_count = labels.len() - positive_count;
    let two = F::one() + F::one();

    let mut positive_rank_sum = F::zero();
    let mut start = 0;
    while start < order.len() {
        let score = scores[order[start]];
        let mut end = start;
        while end < order.len() && scores[order[end]] == score {
            end += 1;
        }
        let lowest_rank = num_traits::cast::<usize, F>(start + 1).expect("ranks fit in the scalar");
        let highest_rank = num_traits::cast::<usize, F>(end).expect("ranks fit in the scalar");
        let average_rank = (lowest_rank + highest_rank) / two;
        for position in start..end {
            if labels[order[position]] == positive_label {
                positive_rank_sum = positive_rank_sum + average_rank;
            }
        }
        start = end;
    }

    let positives = num_traits::cast::<usize, F>(positive_count).expect("counts fit in the scalar");
    let negatives = num_traits::cast::<usize, F>(negative_count).expect("counts fit in the scalar");
    let tie_correction = positives * (positives + F::one()) / two;
    (positive_rank_sum - tie_correction) / (positives * negatives)
}
