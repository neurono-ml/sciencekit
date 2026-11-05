//! Per-class precision, recall and F1.

use sciencekit_common::SKFloat;

use super::confusion::SKConfusionMatrix;

/// Per-class precision, recall and F1, aligned with the confusion-matrix labels.
#[derive(Debug, Clone, PartialEq)]
pub struct SKClassScores<F> {
    precision: Vec<F>,
    recall: Vec<F>,
    f1: Vec<F>,
}

impl<F> SKClassScores<F> {
    /// The precision of each class, in label order.
    pub fn precision(&self) -> &[F] {
        &self.precision
    }

    /// The recall of each class, in label order.
    pub fn recall(&self) -> &[F] {
        &self.recall
    }

    /// The F1 of each class, in label order.
    pub fn f1(&self) -> &[F] {
        &self.f1
    }
}

/// Compute precision, recall and F1 for every class of the confusion matrix.
///
/// A zero denominator yields zero (never NaN); F1 is the harmonic mean of
/// precision and recall and is zero when both are zero.
pub fn sk_per_class_precision_recall_f1<F: SKFloat>(
    matrix: &SKConfusionMatrix,
) -> SKClassScores<F> {
    let classes = matrix.number_of_classes();
    let two = F::one() + F::one();
    let mut precision: Vec<F> = Vec::with_capacity(classes);
    let mut recall: Vec<F> = Vec::with_capacity(classes);
    let mut f1: Vec<F> = Vec::with_capacity(classes);
    for index in 0..classes {
        let true_positive = matrix.true_positive(index);
        let class_precision = sk_ratio::<F>(true_positive, matrix.prediction_count(index));
        let class_recall = sk_ratio::<F>(true_positive, matrix.support(index));
        let denominator = class_precision + class_recall;
        let class_f1 = if denominator.is_zero() {
            F::zero()
        } else {
            two * class_precision * class_recall / denominator
        };
        precision.push(class_precision);
        recall.push(class_recall);
        f1.push(class_f1);
    }
    SKClassScores {
        precision,
        recall,
        f1,
    }
}

/// A ratio that yields zero on a zero denominator.
fn sk_ratio<F: SKFloat>(numerator: usize, denominator: usize) -> F {
    if denominator == 0 {
        return F::zero();
    }
    let numerator = num_traits::cast::<usize, F>(numerator).expect("counts fit in the scalar");
    let denominator =
        num_traits::cast::<usize, F>(denominator).expect("counts fit in the scalar");
    numerator / denominator
}
