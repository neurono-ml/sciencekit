//! Multiclass one-vs-rest ROC/AUC.
//!
//! Each class is scored against all the others with the binary rank-sum AUC.
//! Macro averages the per-class scores equally, weighted by true support and
//! micro ranks every `(observation, class)` pair in one pass. One-vs-one is a
//! documented non-goal of this change.

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};

use super::SKRankingAveraging;
use super::inputs::sk_validate_multiclass_inputs;
use super::roc::sk_rank_auc;

/// The binary one-vs-rest AUC of every class, in ascending label order.
pub fn sk_one_versus_rest_roc_auc_scores<F: SKFloat>(
    scores: &ArrayView2<'_, F>,
    labels: &[i64],
) -> Result<Vec<(i64, F)>, SKError> {
    let classes = sk_validate_multiclass_inputs(scores, labels, "one_versus_rest_roc_auc")?;
    let mut binary_labels = Vec::with_capacity(labels.len());
    let mut class_scores = Vec::with_capacity(labels.len());
    let mut per_class = Vec::with_capacity(classes.len());
    for (column, class_label) in classes.iter().enumerate() {
        binary_labels.clear();
        class_scores.clear();
        for (row, label) in labels.iter().enumerate() {
            binary_labels.push(if *label == *class_label { 1_i64 } else { 0 });
            class_scores.push(scores[[row, column]]);
        }
        per_class.push((*class_label, sk_rank_auc(&class_scores, &binary_labels, 1)));
    }
    Ok(per_class)
}

/// The multiclass one-vs-rest ROC-AUC under the chosen averaging policy.
pub fn sk_multiclass_roc_auc_score<F: SKFloat>(
    scores: &ArrayView2<'_, F>,
    labels: &[i64],
    averaging: SKRankingAveraging,
) -> Result<F, SKError> {
    let per_class = sk_one_versus_rest_roc_auc_scores(scores, labels)?;
    match averaging {
        SKRankingAveraging::Macro => {
            let total = per_class.iter().fold(F::zero(), |sum, (_, auc)| sum + *auc);
            let count =
                num_traits::cast::<usize, F>(per_class.len()).expect("counts fit in the scalar");
            Ok(total / count)
        }
        SKRankingAveraging::Weighted => {
            let observations =
                num_traits::cast::<usize, F>(labels.len()).expect("counts fit in the scalar");
            let weighted = per_class.iter().fold(F::zero(), |sum, (class_label, auc)| {
                let support = labels.iter().filter(|label| *label == class_label).count();
                let support =
                    num_traits::cast::<usize, F>(support).expect("counts fit in the scalar");
                sum + *auc * support
            });
            Ok(weighted / observations)
        }
        SKRankingAveraging::Micro => {
            let classes: Vec<i64> = per_class.iter().map(|(label, _)| *label).collect();
            let mut flat_scores = Vec::with_capacity(labels.len() * classes.len());
            let mut flat_labels = Vec::with_capacity(labels.len() * classes.len());
            for (row, label) in labels.iter().enumerate() {
                for (column, class_label) in classes.iter().enumerate() {
                    flat_scores.push(scores[[row, column]]);
                    flat_labels.push(if label == class_label { 1_i64 } else { 0 });
                }
            }
            Ok(sk_rank_auc(&flat_scores, &flat_labels, 1))
        }
    }
}
